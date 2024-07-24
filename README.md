##TODO:

- [x] Generate prepare small queries ( 20k )
- [x] Generate and prepare medium queries ( 100k )
- [x] Generate and prepare all queries ( 400k )
- [x] Save model weights, with base graph layer separate from cardinality estimation head
- [ ] IMPORTANT: Investigate weird loss sizes, in TEST there is more losses than predictions / actual
- [ ] Generate queries with less repeating predicates (more diversity in predicates and shapes) and hopefully less massive result sets
- [ ] Save different query types separately
- [ ] Save predictions made by model in epoch (Save as: [Query: {pred, actual, q-error}, ..]), seperated by train, 
val, and test files. As we have now, save without query 
- [ ] Create code for experiment visualization, analysis etc. Violin plots, q-error plots with distribution, epoch development etc.
- [ ] Load model weights
- [ ] Use single passed around loss function defined in main pretrain function
- [x] Test performance cardinality estimation on completely unseen benchmark queries
- [ ] Test impact of number of queries on unseen benchmark queries
- [x] Find number of parameters in our model
- [ ] Use cardinality estimator for join order optimization
- [ ] Create system for easy configs for experiment running
- [ ] Generate LUBM benchmark data and test on that
- [ ] Move query generation + execution to this repository and save path, subject star, object star, complex queries separately for more fine-grained analysis.
- [ ] Move RDF2Vec to this repository
- [ ] Use pyRDF2Vec on query entities instead of on all entities in graph (for scalability), like in paper recently published
- [ ] Use the samplers from pyRDF2Vec
- [ ] Parallel random walking?

Ideas:

- Try to use pretraining with deep Q-learning
- Infer RDF2Vec embeddings when new ones arrive by taking random walks starting from this embedding and taking average of the random walk embeddings.
You could even train a model that reconstructs embeddings given random walks, so randomly _corrupt_ embedding, then give it random walks and apply model on random walks so it regenerates the embedding.
- Imitation learning from pretrained cardinality estimator. Reasoning: we copy the cardinality estimator in our weights, but as we get better at imitating we also start including more exploration.
  This exploration might help us find better join plans as we are not bound to possibly inefficient cardinality estimations
  (meaning some accuracy might be redundant, while others might be very necessary).