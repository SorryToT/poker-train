"""다인원(2~9명) 노리밋 홀덤 엔진

카드/족보/버킷 계산은 holdem_engine.py 를 그대로 쓰고, 베팅 규칙만 여러 명용으로 만든 파일이에요.

자리 번호 (포지션 순서)
    0 = SB, 1 = BB, 2 = UTG, ..., 마지막 = BTN(딜러 버튼)
    (2명일 때는 0 = SB 겸 버튼, 1 = BB)
- 프리플랍: BB 다음 사람(UTG) 이 먼저. 2명이면 SB 가 먼저
- 플랍부터: SB 쪽부터(살아 있는 사람 중 가장 앞 번호). 2명이면 BB 가 먼저
- 모두 200칩(100bb) 으로 시작, 매 판 리셋. 블라인드 SB 1 / BB 2

스택이 모두 같아서 '사이드 팟' 이 생기지 않아요 (올인 금액이 모두 200칩으로 같기 때문)

행동 추상화 (AI 가 생각하는 행동)
    f 폴드, c 체크/콜, h ½팟, p 팟, a 올인
- 스트리트 첫 베팅(프리플랍은 첫 레이즈) 은 ½팟/팟, 그 뒤 리레이즈는 팟만
- 스트리트당 레이즈 최대 MAX_RAISES 번, 기본 2 (그 뒤로는 콜/폴드/올인)

게임 상태 (튜플):
(스트리트, 현재 차례, 행동기록, 투입칩 튜플, 폴드 여부 튜플, 이번 스트리트 레이즈 수, 직전 레이즈 크기, 남은 행동 수)
- 남은 행동 수: 이번 라운드가 끝나려면 앞으로 몇 명이 더 행동해야 하는지. 누가 레이즈하면 다시 늘어남
"""
import os
from holdem_engine import (BOARD_COUNT, STREET_NAMES, bucket_labels, card_str, evaluate,  # noqa: F401
                           hand_name, info_key)
import random

FOLD, CALL, HALF, POT, ALLIN = "f", "c", "h", "p", "a"
STACK = 200
BB = 2
# 스트리트당 레이즈 최대 횟수. 기본 2 (기존 학습 결과와 같은 규칙).
# 실전에 더 가까운 규칙으로 따로 학습/비교하려면 환경변수로 바꿔요. 예) Windows: set MP_MAX_RAISES=4  (0 이면 제한 없음)
# 규칙이 다르면 행동 기록이 달라서 전략 파일을 서로 섞을 수 없어요 -> 파일 이름에 r4 같은 꼬리표(TAG)가 붙어요.
# 0 이면 '제한 없음' (실전과 같음). 팟 크기 레이즈가 이어지면 스택이 금방 차서 올인만 남기 때문에 트리가 무한히 커지지는 않아요.
MAX_RAISES = int(os.environ.get("MP_MAX_RAISES", "2"))
TAG = "" if MAX_RAISES == 2 else ("rfree" if MAX_RAISES == 0 else f"r{MAX_RAISES}")
if MAX_RAISES == 0:
    MAX_RAISES = 99
FRACTIONS = {HALF: 0.5, POT: 1.0}
RAISE_ACTIONS = (HALF, POT, ALLIN)


POSITION_NAMES = {
    2: ["SB/BTN", "BB"],
    3: ["SB", "BB", "BTN"],
    4: ["SB", "BB", "CO", "BTN"],
    5: ["SB", "BB", "UTG", "CO", "BTN"],
    6: ["SB", "BB", "UTG", "HJ", "CO", "BTN"],
    7: ["SB", "BB", "UTG", "LJ", "HJ", "CO", "BTN"],
    8: ["SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN"],
    9: ["SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN"],
}


def position_names(n):
    """자리 번호 -> 포지션 이름"""
    return POSITION_NAMES[n]


def deal(n):
    """n 명에게 2장씩 + 보드 5장"""
    d = random.sample(range(52), 2 * n + 5)
    holes = [tuple(d[2 * i:2 * i + 2]) for i in range(n)]
    return holes, tuple(d[2 * n:])


def hand_ranks(holes, board):
    """쇼다운용 패 강도 (판마다 한 번만 계산)"""
    b = list(board)
    return [evaluate(list(h) + b) for h in holes]


def start_state(n):
    contribs = [0] * n
    contribs[0], contribs[1] = 1, 2
    first = 0 if n == 2 else 2
    return (0, first, "", tuple(contribs), (False,) * n, 0, BB, n)


def _can_act(state, i):
    return not state[4][i] and state[3][i] < STACK


def _next_actor(contribs, folded, after):
    n = len(contribs)
    for k in range(1, n + 1):
        i = (after + k) % n
        if not folded[i] and contribs[i] < STACK:
            return i
    return None


def raise_to(state, action):
    _, p, _, contribs, _, _, last, _ = state
    if action == ALLIN:
        return STACK
    high = max(contribs)
    pot_after_call = sum(contribs) + (high - contribs[p])
    by = max(int(round(FRACTIONS[action] * pot_after_call)), last, BB)
    return min(STACK, high + by)


def legal_actions(state):
    _, p, _, contribs, _, raises, _, _ = state
    high = max(contribs)
    mine = contribs[p]
    actions = []
    if high > mine:
        actions.append(FOLD)
    actions.append(CALL)
    if high < STACK:
        if raises < MAX_RAISES:
            sizes = (HALF, POT) if raises == 0 else (POT,)
            seen = set()
            for a in sizes:
                amt = raise_to(state, a)
                if amt < STACK and amt not in seen:
                    seen.add(amt)
                    actions.append(a)
        actions.append(ALLIN)
    return actions


def step(state, action, ranks, amount=None):
    """행동 적용.
    ranks: 각 자리의 쇼다운 패 강도 리스트 (hand_ranks 결과)
    amount: 사람이 자유 금액으로 레이즈할 때의 실제 총 투입칩
    반환: (다음 상태, None) 또는 게임 종료 시 (None, 각 자리의 수익 리스트)
    """
    street, p, history, contribs, folded, raises, last, pending = state
    contribs = list(contribs)
    folded = list(folded)
    n = len(contribs)
    high = max(contribs)

    if action == FOLD:
        folded[p] = True
        pending -= 1
        alive = [i for i in range(n) if not folded[i]]
        if len(alive) == 1:                       # 한 명만 남음 -> 그 사람이 팟을 가져감
            return None, _payout(contribs, alive)
    elif action == CALL:
        contribs[p] = high
        pending -= 1
    else:
        new = amount if amount is not None else raise_to(state, action)
        last = max(last, new - high)
        contribs[p] = new
        raises += 1
        # 레이즈한 사람을 뺀 나머지 중 행동할 수 있는 사람은 다시 행동해야 함
        pending = sum(1 for i in range(n) if i != p and not folded[i] and contribs[i] < STACK)

    history += action

    if pending <= 0:     # 이번 라운드 끝
        can_act = [i for i in range(n) if not folded[i] and contribs[i] < STACK]
        if street == 3 or len(can_act) <= 1:      # 리버 끝, 또는 더 베팅할 사람이 없음(올인) -> 쇼다운
            return None, _showdown(contribs, folded, ranks)
        start_from = 0 if n > 2 else 1
        first = _next_actor(contribs, folded, start_from - 1)
        return (street + 1, first, history + "/", tuple(contribs), tuple(folded), 0, BB, len(can_act)), None

    nxt = _next_actor(contribs, folded, p)
    return (street, nxt, history, tuple(contribs), tuple(folded), raises, last, pending), None


def _payout(contribs, winners):
    pot = sum(contribs)
    out = [-c for c in contribs]
    share = pot / len(winners)
    for w in winners:
        out[w] += share
    return out


def _showdown(contribs, folded, ranks):
    alive = [i for i in range(len(contribs)) if not folded[i]]
    best = max(ranks[i] for i in alive)
    return _payout(contribs, [i for i in alive if ranks[i] == best])


# ---------------------------------------------------------------------------
# 사람의 자유 금액 지원 (노리밋 엔진과 같은 방식)
# ---------------------------------------------------------------------------
def min_raise_to(state):
    _, _, _, contribs, _, _, last, _ = state
    return min(STACK, max(contribs) + max(last, BB))


def raise_fraction(state, amount):
    _, p, _, contribs, _, _, _, _ = state
    high = max(contribs)
    pot_after_call = sum(contribs) + (high - contribs[p])
    return (amount - high) / pot_after_call


def translate(state, amount, rng):
    """사람 레이즈 총액 -> AI 가 아는 크기 문자 (무작위 의사조화 매핑)"""
    if amount >= STACK:
        return ALLIN
    x = raise_fraction(state, amount)
    options = sorted((raise_fraction(state, raise_to(state, a)), a)
                     for a in legal_actions(state) if a in RAISE_ACTIONS)
    if x <= options[0][0]:
        return options[0][1]
    if x >= options[-1][0]:
        return options[-1][1]
    for (fa, a), (fb, b) in zip(options, options[1:]):
        if fa <= x <= fb:
            if fb == fa:
                return a
            p_low = (fb - x) * (1 + fa) / ((fb - fa) * (1 + x))
            return a if rng.random() < p_low else b
    return options[-1][1]


def replay_abstract(n, history):
    """번역된 행동기록을 학습한 게임 트리에서 다시 따라가 봄. 벗어나면 None"""
    state = start_state(n)
    dummy = [0] * n
    for ch in history.replace("/", ""):
        if ch not in legal_actions(state):
            return None
        state, _ = step(state, ch, dummy)
        if state is None:
            return None
    return state if state[2] == history else None
