# Week 28: classical cont. and GRU

- Not much happened, VPN to Clark not working
    - Since I access stuff over SSHFS don't have much to show

- Analyse results from PCA
    - Esp the -0.733 correlation between 1dim PCA and labels
    - Weights make sense, mostly weighted towards the centre
- GRU increases accuracy slightly, it can make decisions earlier
    - Avg distance < 0.1 at t=80
    - GRU decreases that to around 60
    - GRU specifically trained with t=25 to t=50, 55, or 60
    - Maybe work on a sliding window later
