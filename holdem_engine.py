"""헤즈업 리미트 홀덤 게임 엔진 (파이썬 표준 라이브러리만 사용)

이 파일 하나에 들어 있는 것
1. 카드 표현과 족보(핸드 랭킹) 평가
2. 리미트 홀덤 베팅 규칙 (legal_actions, step) - Leduc 엔진과 같은 모양
3. 카드 추상화(버킷): 비슷한 강도의 패를 한 묶음으로 취급해서 학습량을 줄임

규칙 요약
- 플레이어 0 = 스몰 블라인드(SB, 딜러 버튼), 플레이어 1 = 빅 블라인드(BB)
- 블라인드: SB 1칩, BB 2칩
- 베팅 단위: 프리플랍/플랍 2칩, 턴/리버 4칩 (리미트라서 크기가 고정)
- 한 스트리트에 베팅+레이즈는 최대 4번 (프리플랍은 BB 블라인드가 첫 번째 베팅으로 계산됨)
- 프리플랍은 SB 가 먼저, 플랍부터는 BB 가 먼저 행동
- 행동: f = 폴드, c = 콜/체크, r = 베팅/레이즈
"""
import random

FOLD, CALL, RAISE = "f", "c", "r"
BET_SIZE = [2, 2, 4, 4]          # 스트리트별 베팅 단위 (프리플랍, 플랍, 턴, 리버)
MAX_BETS = 4                     # 스트리트당 베팅+레이즈 최대 횟수
STREET_NAMES = ["프리플랍", "플랍", "턴", "리버"]
BOARD_COUNT = [0, 3, 4, 5]       # 스트리트별 공개된 보드 카드 수

# ---------------------------------------------------------------------------
# 1. 카드
# ---------------------------------------------------------------------------
# 카드는 0~51 숫자. 숫자 // 4 = 랭크(0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A), 숫자 % 4 = 무늬
RANKS = "23456789TJQKA"
SUITS = "shdc"            # 스페이드, 하트, 다이아, 클럽
SUIT_SYMBOLS = "♠♥♦♣"


def card_str(c):
    """카드 숫자 -> 'As', 'Td' 같은 문자열"""
    return RANKS[c >> 2] + SUITS[c & 3]


# ---------------------------------------------------------------------------
# 2. 족보 평가
# ---------------------------------------------------------------------------
# 결과는 정수 하나. 숫자가 클수록 강한 패라서 그냥 크기 비교만 하면 됨.
# 족보 번호: 8 스트레이트 플러시, 7 포카드, 6 풀하우스, 5 플러시, 4 스트레이트,
#            3 트리플, 2 투페어, 1 원페어, 0 하이카드
HAND_NAMES = ["하이카드", "원페어", "투페어", "트리플", "스트레이트",
              "플러시", "풀하우스", "포카드", "스트레이트 플러시"]


def _build_straight_table():
    """랭크 비트마스크(13비트) -> 가장 높은 스트레이트의 꼭대기 랭크 (없으면 -1)"""
    table = [-1] * 8192
    for mask in range(8192):
        for top in range(12, 3, -1):           # A-high 부터 6-high 까지
            need = 0b11111 << (top - 4)
            if mask & need == need:
                table[mask] = top
                break
        else:
            wheel = (1 << 12) | 0b1111          # A-2-3-4-5 (휠)
            if mask & wheel == wheel:
                table[mask] = 3                  # 5 가 꼭대기
    return table


STRAIGHT_TOP = _build_straight_table()


def _top_ranks(mask, n):
    """비트마스크에서 높은 랭크 n 개를 골라 리스트로"""
    out = []
    r = 12
    while r >= 0 and len(out) < n:
        if mask >> r & 1:
            out.append(r)
        r -= 1
    return out


def _encode(category, kickers):
    v = category
    for k in kickers:
        v = v * 16 + k
    for _ in range(5 - len(kickers)):
        v *= 16
    return v


def evaluate(cards):
    """5~7장 카드의 패 강도를 정수로 반환 (클수록 강함)"""
    counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suit_masks[c & 3] |= 1 << r

    # 플러시가 있는지 확인 (스트레이트 플러시는 바로 반환)
    flush_mask = 0
    for m in suit_masks:
        if bin(m).count("1") >= 5:
            top = STRAIGHT_TOP[m]
            if top >= 0:
                return _encode(8, [top])
            flush_mask = m

    quads, trips, pairs, singles = [], [], [], []
    all_mask = 0
    for r in range(12, -1, -1):
        n = counts[r]
        if n:
            all_mask |= 1 << r
            if n == 4:
                quads.append(r)
            elif n == 3:
                trips.append(r)
            elif n == 2:
                pairs.append(r)
            else:
                singles.append(r)

    if quads:
        q = quads[0]
        kicker = _top_ranks(all_mask & ~(1 << q), 1)
        return _encode(7, [q] + kicker)
    if trips and (len(trips) >= 2 or pairs):
        t = trips[0]
        p = max(trips[1] if len(trips) >= 2 else -1, pairs[0] if pairs else -1)
        return _encode(6, [t, p])
    if flush_mask:      # 포카드/풀하우스보다 약하고 스트레이트보다 강함
        return _encode(5, _top_ranks(flush_mask, 5))
    top = STRAIGHT_TOP[all_mask]
    if top >= 0:
        return _encode(4, [top])
    if trips:
        t = trips[0]
        return _encode(3, [t] + _top_ranks(all_mask & ~(1 << t), 2))
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        return _encode(2, [p1, p2] + _top_ranks(all_mask & ~(1 << p1) & ~(1 << p2), 1))
    if pairs:
        p = pairs[0]
        return _encode(1, [p] + _top_ranks(all_mask & ~(1 << p), 3))
    return _encode(0, _top_ranks(all_mask, 5))


def hand_name(cards):
    return HAND_NAMES[evaluate(cards) >> 20]


def showdown_winner(hole0, hole1, board):
    """쇼다운 승자: 0 또는 1, 무승부면 -1"""
    s0 = evaluate(hole0 + board)
    s1 = evaluate(hole1 + board)
    if s0 > s1:
        return 0
    if s1 > s0:
        return 1
    return -1


def deal():
    """카드 9장을 섞어서 (플레이어0 패, 플레이어1 패, 보드 5장) 으로 반환"""
    d = random.sample(range(52), 9)
    return (d[0], d[1]), (d[2], d[3]), tuple(d[4:9])


# ---------------------------------------------------------------------------
# 3. 베팅 규칙
# ---------------------------------------------------------------------------
# 게임 상태 (Leduc 과 같은 모양):
# (스트리트, 현재 차례, 행동기록, 플레이어0 투입칩, 플레이어1 투입칩, 이번 스트리트 베팅 수, 이번 스트리트 행동 수)
# 행동기록에서 스트리트가 바뀔 때마다 "/" 가 들어감. 예: "rc/cr" = 프리플랍 레이즈-콜, 플랍 체크-베팅
START = (0, 0, "", 1, 2, 1, 0)


def legal_actions(state):
    _, player, _, c0, c1, bets, _ = state
    mine, theirs = (c0, c1) if player == 0 else (c1, c0)
    actions = []
    if theirs > mine:            # 상대가 더 많이 냈을 때만 폴드 가능
        actions.append(FOLD)
    actions.append(CALL)
    if bets < MAX_BETS:
        actions.append(RAISE)
    return actions


def step(state, action, winner):
    """행동을 적용.
    winner = 쇼다운까지 갔을 때의 승자 (0, 1, 무승부 -1). 판마다 한 번만 계산해서 넘겨주면 됨.
    반환: (다음 상태, None) 또는 게임 종료 시 (None, 플레이어0 기준 수익)
    """
    street, player, history, c0, c1, bets, acted = state
    contrib = [c0, c1]
    opp = 1 - player

    if action == FOLD:
        return None, (-contrib[0] if player == 0 else contrib[1])

    facing = contrib[opp] > contrib[player]
    round_over = False
    if action == CALL:
        contrib[player] = contrib[opp]
        if facing:
            # 예외: 프리플랍에 SB 가 처음 콜(림프)하면 BB 에게 체크/레이즈 기회가 남아 있음
            round_over = not (street == 0 and acted == 0)
        else:
            round_over = acted >= 1    # 체크-체크
    else:  # RAISE
        contrib[player] = contrib[opp] + BET_SIZE[street]
        bets += 1

    history += action
    acted += 1

    if round_over:
        if street < 3:   # 다음 스트리트: BB(플레이어 1) 가 먼저 행동
            return (street + 1, 1, history + "/", contrib[0], contrib[1], 0, 0), None
        if winner == 0:
            return None, contrib[1]
        if winner == 1:
            return None, -contrib[0]
        return None, 0
    return (street, opp, history, contrib[0], contrib[1], bets, acted), None


# ---------------------------------------------------------------------------
# 4. 카드 추상화 (버킷)
# ---------------------------------------------------------------------------
# 홀덤은 카드 조합이 너무 많아서(플랍만 해도 약 2만 5천 가지 x 패) 그대로는 학습할 수 없음.
# 그래서 "지금 내 패가 얼마나 강한가"를 0~1 숫자(승률)로 계산한 뒤,
# 비슷한 승률끼리 같은 버킷 번호로 묶어서 학습함.
# - 프리플랍: 169 가지 시작패 종류 그대로 사용 (예: AKs = A,K 같은 무늬 / 72o = 다른 무늬)
# - 플랍/턴/리버: 무작위 상대 패와 남은 보드를 여러 번 뽑아서 승률을 추정 -> 버킷 번호
POSTFLOP_BUCKETS = 10


def preflop_class(hole):
    r1, r2 = hole[0] >> 2, hole[1] >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if (hole[0] & 3) == (hole[1] & 3) else "o")


def equity(hole, board, samples):
    """무작위 상대 패를 상대로 한 승률(무승부는 0.5) 을 몬테카를로로 추정"""
    known = set(hole) | set(board)
    rest = [c for c in range(52) if c not in known]
    need = 5 - len(board)
    hole = list(hole)
    board = list(board)
    score = 0.0
    if need == 0:                            # 리버: 내 패 강도는 한 번만 계산
        mine = evaluate(hole + board)
        for _ in range(samples):
            o = random.sample(rest, 2)
            theirs = evaluate(o + board)
            score += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        return score / samples
    for _ in range(samples):
        draw = random.sample(rest, 2 + need)
        full = board + draw[2:]
        mine = evaluate(hole + full)
        theirs = evaluate(draw[:2] + full)
        score += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return score / samples


def bucket_labels(hole, board, samples=40, version=1):
    """스트리트별(0~3) 버킷 이름 4개를 한 번에 계산. 정보집합 키의 앞부분으로 사용
    version=1: 승률만 (예: '6')
    version=2: 플랍/턴에서 '앞으로 좋아질 패인지' 까지 (예: '6+' 드로우, '6-' 따라잡히기 쉬운 메이드, '6' 보통)
    """
    labels = [preflop_class(hole)]
    for street in (1, 2, 3):
        b = board[:BOARD_COUNT[street]]
        if version == 2 and street < 3:
            now, final = equity_now_and_final(hole, b, samples)
            tag = "+" if final - now > 0.08 else ("-" if final - now < -0.08 else "")
            labels.append(str(min(POSTFLOP_BUCKETS - 1, int(final * POSTFLOP_BUCKETS))) + tag)
        else:
            eq = equity(hole, b, samples)
            labels.append(str(min(POSTFLOP_BUCKETS - 1, int(eq * POSTFLOP_BUCKETS))))
    return labels


def equity_now_and_final(hole, board, samples):
    """(지금 보드 기준 승률, 끝까지 갔을 때 승률) 을 같은 표본으로 함께 추정
    - 지금 승률이 낮은데 끝 승률이 높으면: 드로우 (플러시/스트레이트를 기다리는 패)
    - 지금 승률이 높은데 끝 승률이 낮으면: 지금은 이기지만 따라잡히기 쉬운 패"""
    known = set(hole) | set(board)
    rest = [c for c in range(52) if c not in known]
    need = 5 - len(board)
    hole, board = list(hole), list(board)
    mine_now = evaluate(hole + board)
    now = final = 0.0
    for _ in range(samples):
        draw = random.sample(rest, 2 + need)
        opp = draw[:2]
        theirs_now = evaluate(opp + board)
        now += 1.0 if mine_now > theirs_now else (0.5 if mine_now == theirs_now else 0.0)
        full = board + draw[2:]
        m, t = evaluate(hole + full), evaluate(opp + full)
        final += 1.0 if m > t else (0.5 if m == t else 0.0)
    return now / samples, final / samples


def info_key(label, history):
    """정보집합 키: '버킷|행동기록'. 행동기록의 '/' 개수로 스트리트를 알 수 있음"""
    return label + "|" + history
