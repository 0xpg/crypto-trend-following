# Source thread evidence map

Root post: [1587591552691765251](https://x.com/i/web/status/1587591552691765251)

Published: 2022-11-01 and 2022-11-02

Retrieved: 2026-07-26

This file is a neutral paraphrase of the 37-post construction thread. Post
numbers and identifiers preserve the evidence trail without reproducing the
source voice. Punctuation is ASCII.

## Posts 1-4: setting

1. Post 1587591552691765251 presents trend following as a direct systematic
   rule: buy rising markets and sell falling markets.
2. Post 1587591555577221120 describes the scale of the CTA industry and the
   presence of internal implementations at multi-strategy funds.
3. Post 1587591558333095936 notes experience with both simple and complex
   implementations across a wide range of asset sizes.
4. Post 1587591560773963778 takes the existence of exploitable trends as the
   premise and limits the discussion to strategy construction.

## Posts 5-9: components and universe

5. Post 1587591563173142528 lists seven components: universe selection,
   trend detection, signal mapping, sector weights, portfolio risk targeting,
   trading rules and execution.
6. Post 1587591565819666432 recommends as many diversifying markets as
   possible. It gives about 0.2 Sharpe for one market and states square-root
   scaling in the number of independent markets.
7. Post 1587591568197828611 contrasts another correlated equity index with an
   uncorrelated natural-gas contract. The former adds little; the latter adds
   meaningful diversification.
8. Post 1587591570806960128 gives 60-80 major futures markets as a typical
   range and 300-400 markets for a larger program.
9. Post 1587591573491306496 extends the universe to niche derivatives and
   estimates a 1.5-2x Sharpe gain from broader coverage.

## Posts 10-18: trend signal and response

10. Post 1587591575974068231 identifies moving-average crossovers as the most
    common trend signal and emphasizes a quantitative directional measure.
11. Post 1587591578331250691 specifies fast minus slow moving averages of log
    price, normalized by volatility. The construction acts as a low-pass
    filter and leaves little room for parameter fitting.
12. Post 1587591580961198080 gives a horizon range of one month to one year,
    with three to six months as a typical average.
13. Post 1587591583351857152 maps trend strength to target market risk with a
    response that stops growing at extreme signals.
14. Post 1587591587722559489 gives two response families: a sigmoid and an
    overextension function proportional to `x*exp(-x^2)`.
15. Post 1587591590922567683 marks a pause before sector weights, portfolio
    risk targeting and execution.
16. Post 1587592131845160963 corrects a typo in the opening post.
17. Post 1587719216681172992 discusses an optional long bias for assets with
    positive expected drift and notes the loss of crisis symmetry.
18. Post 1587719480163139587 observes that many trend programs still use such
    a bias.

## Posts 19-24: sizing and sectors

19. Post 1587719915875799040 describes inverse-volatility sizing as the most
    common way to give each market similar risk at a fixed signal strength.
20. Post 1587720612306460672 recommends a simple volatility forecast blending
    a long-run estimate with 30-60 day realized volatility.
21. Post 1587721985974587393 introduces sector weights as protection against
    overexposure to an asset class with many listed contracts.
22. Post 1587722517116067841 gives examples involving unequal market counts
    and unequal within-sector correlations.
23. Post 1587722893814861825 proposes scaling sectors toward equal long-run
    risk.
24. Post 1587723744407175169 notes methods from equal weighting to hierarchical
    models and favors broad asset-class buckets with similar long-run risk.

## Posts 25-30: portfolio risk

25. Post 1587725558850486278 states that an untargeted portfolio can range
    from 0.3x to 3x its intended risk as the number of trends changes.
26. Post 1587726013370359808 notes that this dispersion may be unsuitable for
    conservative capital.
27. Post 1587726811953303552 scales up sparse-trend books and scales down
    crowded-trend books to stay near the target, with a claimed Sharpe benefit.
28. Post 1587727473797746688 notes the trade-off between consistent risk and
    gains during unusually broad trends.
29. Post 1587727749342527490 makes the preferred risk profile dependent on the
    strategy's role beside other allocations.
30. Post 1587741284684398594 marks a second pause.

## Posts 31-37: trading and execution

31. Post 1587896137918316546 recommends a no-trade buffer around target
    positions to control commissions and slippage.
32. Post 1587896728254124036 states that the buffer keeps holdings near target
    while reducing turnover.
33. Post 1587897113429807104 allows wider buffers for wide spreads, low
    volatility and large positions, with gradual movement toward target for
    large books.
34. Post 1587897592448516096 reserves portfolio optimizers for cases that need
    them and otherwise favors simple heuristics.
35. Post 1587898062168625154 separates in-house execution, broker execution,
    algorithms and manual execution.
36. Post 1587898336983470081 regards algorithms as adequate for most liquid
    markets and manual handling as useful for less liquid contracts.
37. Post 1587898574817447937 closes the construction thread.
