def compute_cer(hyps: list[str], refs: list[str]) -> float:
    """Character error rate on digit strings (e.g. '139473' vs '139000')."""
    if not refs:
        return 0.0
    total_dist = sum(_edit_distance(h, r) for h, r in zip(hyps, refs))
    total_len = sum(len(r) for r in refs)
    return total_dist / max(total_len, 1)


def harmonic_mean_cer(cer_ind: float, cer_ood: float) -> float:
    denom = cer_ind + cer_ood
    return 2.0 * cer_ind * cer_ood / denom if denom > 0 else 0.0


def _edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
