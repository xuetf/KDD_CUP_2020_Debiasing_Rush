# Solution of KDD CUP 2020 (Debiasing) of Team Rush
Track B,

- ndcg-full: **rank 3rd**（7.277)
- ndcg-half: **rank 10th** (7.226)


## Components of the solution
python 3.6
- Recall
    - Item-CF
    - User-CF
    - Swing
    - Bi-Graph
    - SR-GNN
    
- Ranking
    - LGB
    - DIN
    
## Running
- ./run.sh

We will release the code in github soon. The code is not well checked because of the strict deadline.
We will double-check the code as soon as possible.
    
## References

[1]  Wu S, Tang Y, Zhu Y, et al. Session-based recommendation with graph neural networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 346-353.

[2]  Gupta P, Garg D, Malhotra P, et al. NISER: Normalized Item and Session Representations with Graph Neural Networks[J]. arXiv preprint arXiv:1909.04276, 2019.

[3]  Zhou T, Ren J, Medo M, et al. Bipartite network projection and personal recommendation[J]. Physical review E, 2007, 76(4): 046115.

[4] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018: 1059-1068.

[5] Ke G, Meng Q, Finley T, et al. Lightgbm: A highly efficient gradient boosting decision tree[C]//Advances in neural information processing systems. 2017: 3146-3154.

[6] DeepCTR, Easy-to-use,Modular and Extendible package of deep-learning based CTR models, https://github.com/shenweichen/DeepCTR

[7] A simple itemCF Baseline, score:0.1169(phase0-2), https://tianchi.aliyun.com/forum/postDetail?postId=103530

[8] 改进青禹小生baseline，phase3线上0.2, https://tianchi.aliyun.com/forum/postDetail?postId=105787

[9] 推荐系统算法调研, http://xtf615.com/2018/05/03/recommender-system-survey/

[10] A Simple Recall Method based on Network-based Inference, score:0.18 (phase0-3), https://tianchi.aliyun.com/forum/postDetail?postId=104936

[11] A library for efficient similarity search and clustering of dense vectors, https://github.com/facebookresearch/faiss

    

