# Solution of KDD CUP 2020 (Debiasing) of Team Rush
Track B,

- ndcg-full: **rank 3rd**（7.277)
- ndcg-half: **rank 10th** (7.226)

## Key Points of this Solution 
- The construction of the recall training set, how to use the ENTIRE data for training is important. we need to prevent data crossing from both the user side and the item side. This improvement is very significant, indicating that the data have a great impact on the results.

- Improvements in CF methods can effectively debias the data, including interaction time, direction, content similarity, item popularity, user activity, etc. This improvement is also very significant and suits the topic of the game, i.e., Debiasing.

- SR-GNN is designed for sequential recommendation based on the graph neural networks, which perfectly suits the scenario of this competition. SR-GNN captures the high-order proximity between items and takes into account the user's long-term and short-term preferences. In addition, we improve SR-GNN in many ways, i.e., we use content features for embedding initialization, introducing node weights according to frequency (for debiasing), position embedding (enhance short-term interaction importance), embedding L2 normalization, residual connection, the sequence-level embedding construction. The improvement of the SR-GNN reaches more than 0.05+ in terms of full-ndcg.

- The coarse-grained ranking considers the frequency of items and motivate the exposure of low-frequency items to effectively debias, which significantly improves the half metrics.

- Construction of ranking features, including recall features, content features, historical behavior-related features, ID features, etc.

- Ensemble model: the fusion of Tree-based model, i.e., GBDT and Deep-Learning-based model, i.e., DIN can effectively improve the ranking results

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

The recall code is well checked.
The ranking code is not well checked now, we will double-check the code as soon as possible.
    
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

    

