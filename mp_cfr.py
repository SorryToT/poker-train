"""다인원(2~9명) 노리밋 홀덤 MCCFR 학습

실행
    python mp_cfr.py 3 60      # 3인 테이블을 60분 학습
    python mp_cfr.py 6 60      # 6인 테이블
    python mp_cfr.py 9 0       # 9인: 학습 없이 평가만

- 인원별로 파일이 따로 생겨요 (MP_MAX_RAISES 를 바꾸면 mp6r4_... 처럼 꼬리표가 붙어요): mp3_strategy.json / mp3_checkpoint.pkl, mp6_..., mp9_...
- 다시 실행하면 이어서 학습, 5분마다 자동 저장, Ctrl+C 로 멈춰도 저장

다인원 MCCFR (Pluribus 와 같은 External Sampling 방식)
- 한 판마다 '학습하는 사람(traverser)' 을 한 명씩 돌아가며 정하고
- 그 사람 차례에서는 모든 행동을 다 시험해서 후회값을 계산,
- 다른 사람들 차례에서는 현재 전략대로 하나씩 뽑아서 진행
[2026-11-19] 학습 개선 (Pluribus 에서 가져옴)
- Linear MCCFR: 나중 반복일수록 큰 가중치로 쌓아요 (초반에 대충 배운 값의 영향을 점점 줄임)
  예전 체크포인트를 처음 불러오면, 지금까지 쌓인 값을 '평균 가중치' 로 한 번 바꿔서 이어가요 (linear 표시가 붙음)
- 가지치기(pruning): 후회값이 아주 나쁜 행동은 95% 확률로 탐색을 건너뜀 (리버와 폴드는 항상 탐색)
  -> 같은 시간에 더 많은 판을 학습
- --no-report: 학습 후 평가를 건너뜀 (여러 개를 동시에 돌릴 때)

"""
import json
import os
import pickle
import random
import sys
import time

from mp_engine import (ALLIN, CALL, POT, TAG, bucket_labels, deal, hand_ranks, info_key, legal_actions,
                       start_state, step)

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_EVERY = 5 * 60

nodes = {}
iterations_done = 0
N_PLAYERS = 3


def paths(n):
    return (os.path.join(HERE, f"mp{n}{TAG}_checkpoint.pkl"), os.path.join(HERE, f"mp{n}{TAG}_strategy.json"))


def current_strategy(regrets):
    pos = [r if r > 0 else 0.0 for r in regrets]
    total = sum(pos)
    k = len(regrets)
    return [x / total for x in pos] if total > 0 else [1.0 / k] * k

# ---------------------------------------------------------------------------
# Linear MCCFR + 가지치기
# ---------------------------------------------------------------------------
LINEAR_UNIT = 10_000          # 이만큼 학습할 때마다 가중치 +1
PRUNE_START = 200_000         # 이만큼 학습한 뒤부터 가지치기
PRUNE_THRESHOLD = 1000.0      # 후회값이 -(이 값 x 현재 가중치) 보다 나쁘면 건너뜀 (칩 단위)
prune_now = False
linear = False


def weight():
    return max(1.0, iterations_done / LINEAR_UNIT)


def to_linear():
    """예전(가중치 없는) 체크포인트 -> Linear 형식으로 한 번 변환.
    지금까지 T판 쌓인 값은 평균 가중치 (T/2)/LINEAR_UNIT 로 쌓였다고 보고 곱해줌"""
    global linear
    scale = max(1.0, iterations_done / 2 / LINEAR_UNIT)
    if scale > 1.0:
        for regrets, strat_sum in nodes.values():
            for i in range(len(regrets)):
                regrets[i] *= scale
                strat_sum[i] *= scale
    linear = True
    print(f"  Linear MCCFR 로 변환 (기존 값 x{scale:.1f})")



def traverse(state, traverser, labels, ranks):
    """traverser 의 기대 수익을 반환"""
    player = state[1]
    actions = legal_actions(state)
    key = info_key(labels[player][state[0]], state[2])
    node = nodes.get(key)
    if node is None:
        k = len(actions)
        node = nodes[key] = [[0.0] * k, [0.0] * k]
    regrets, strat_sum = node
    strategy = current_strategy(regrets)

    if player == traverser:
        values = []
        node_value = 0.0
        w = weight()
        cut = -PRUNE_THRESHOLD * w
        pruning = prune_now and state[0] < 3
        for i, a in enumerate(actions):
            if pruning and regrets[i] < cut and a != "f":
                values.append(None)              # 가지치기: 아주 나쁜 행동은 이번엔 탐색 안 함
                continue
            nxt, payoff = step(state, a, ranks)
            v = payoff[traverser] if nxt is None else traverse(nxt, traverser, labels, ranks)
            values.append(v)
            node_value += strategy[i] * v
        for i in range(len(actions)):
            if values[i] is not None:
                regrets[i] += w * (values[i] - node_value)
        return node_value

    w = weight()
    for i in range(len(actions)):
        strat_sum[i] += w * strategy[i]
    a = random.choices(actions, weights=strategy)[0]
    nxt, payoff = step(state, a, ranks)
    return payoff[traverser] if nxt is None else traverse(nxt, traverser, labels, ranks)


def train_one():
    global prune_now
    prune_now = iterations_done >= PRUNE_START and random.random() < 0.95
    n = N_PLAYERS
    holes, board = deal(n)
    labels = [bucket_labels(h, board) for h in holes]
    ranks = hand_ranks(holes, board)
    start = start_state(n)
    for t in range(n):
        traverse(start, t, labels, ranks)


def train_for(minutes):
    global iterations_done
    end = time.time() + minutes * 60
    t0 = last_print = last_save = time.time()
    start_iter = iterations_done
    while time.time() < end:
        for _ in range(20):
            train_one()
        iterations_done += 20
        now = time.time()
        if now - last_print > 30:
            last_print = now
            print(f"  [{N_PLAYERS}인] {iterations_done:,}판 학습 (이번 실행 {iterations_done - start_iter:,}판, "
                  f"정보집합 {len(nodes):,}개, 남은 시간 {max(0, end - now) / 60:.1f}분)")
        if now - last_save > SAVE_EVERY:
            last_save = now
            save_all()
    print(f"이번 실행: {iterations_done - start_iter:,}판, {time.time() - t0:.0f}초")


def global_linear_on():
    global linear
    linear = True


def load_checkpoint():
    global nodes, iterations_done
    ck, _ = paths(N_PLAYERS)
    if os.path.exists(ck):
        with open(ck, "rb") as f:
            data = pickle.load(f)
        nodes = data["nodes"]
        iterations_done = data["iterations"]
        print(f"이전 학습 불러옴 ({N_PLAYERS}인): {iterations_done:,}판, 정보집합 {len(nodes):,}개")
        if data.get("linear"):
            global linear
            linear = True
        else:
            to_linear()
    else:
        global_linear_on()


def save_all():
    ck, st = paths(N_PLAYERS)
    with open(ck + ".tmp", "wb") as f:
        pickle.dump({"nodes": nodes, "iterations": iterations_done, "players": N_PLAYERS, "linear": linear}, f)
    os.replace(ck + ".tmp", ck)
    strategy = {}
    for key, (_, strat_sum) in nodes.items():
        total = sum(strat_sum)
        if total > 0:
            strategy[key] = [round(s / total, 3) for s in strat_sum]
    with open(st + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"iterations": iterations_done, "players": N_PLAYERS, "strategy": strategy}, f)
    os.replace(st + ".tmp", st)
    print(f"  [저장] {N_PLAYERS}인 {iterations_done:,}판 학습분, 정보집합 {len(nodes):,}개")


# ---------------------------------------------------------------------------
# 평가: AI 한 명 vs 나머지 자리 전부 같은 봇. AI 자리를 돌아가며 앉힘
# ---------------------------------------------------------------------------
def make_cfr_policy(strategy):
    def policy(state, actions, label):
        probs = strategy.get(info_key(label, state[2]))
        if probs is None or len(probs) != len(actions) or sum(probs) <= 0:
            return CALL
        return random.choices(actions, weights=probs)[0]
    return policy


SIMPLE_BOTS = {
    "무작위 봇": lambda s, a, l: random.choice(a),
    "항상 콜 봇": lambda s, a, l: CALL,
    "팟 레이즈 봇": lambda s, a, l: POT if POT in a else (ALLIN if ALLIN in a else CALL),
}


def play_hand(policies):
    n = len(policies)
    holes, board = deal(n)
    labels = [bucket_labels(h, board) for h in holes]
    ranks = hand_ranks(holes, board)
    state = start_state(n)
    while True:
        p = state[1]
        a = policies[p](state, legal_actions(state), labels[p][state[0]])
        state, payoff = step(state, a, ranks)
        if state is None:
            return payoff


def evaluate_vs(ai, other, hands):
    """AI 1명 + 나머지는 other 봇. AI 의 bb/100"""
    n = N_PLAYERS
    total = 0.0
    for i in range(hands):
        seat = i % n
        pols = [other] * n
        pols[seat] = ai
        total += play_hand(pols)[seat]
    return total / hands / 2 * 100


def report(hands=3000):
    _, st = paths(N_PLAYERS)
    with open(st, encoding="utf-8") as f:
        ai = make_cfr_policy(json.load(f)["strategy"])
    print(f"\n평가 ({N_PLAYERS}인 테이블, AI 1명 vs 나머지 {N_PLAYERS - 1}명 같은 봇, {hands}판, 단위 bb/100)")
    for name, bot in SIMPLE_BOTS.items():
        print(f"  AI vs {name}들: {evaluate_vs(ai, bot, hands):+.1f}")


if __name__ == "__main__":
    no_report = "--no-report" in sys.argv
    if no_report:
        sys.argv.remove("--no-report")
    N_PLAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 10
    if not 2 <= N_PLAYERS <= 9:
        sys.exit("인원은 2~9 사이로 주세요. 예: python mp_cfr.py 6 60")
    load_checkpoint()
    if minutes > 0:
        print(f"{N_PLAYERS}인 테이블을 {minutes:g}분 동안 학습합니다. (5분마다 자동 저장)")
        try:
            train_for(minutes)
        except KeyboardInterrupt:
            print("\n중단! 지금까지 학습한 내용을 저장합니다.")
        save_all()
    if os.path.exists(paths(N_PLAYERS)[1]) and not no_report:
        report()
