##Installing requirements:
Needs 

##TODO:

- [x] Generate prepare small queries ( 20k )
- [x] Generate and prepare medium queries ( 100k )
- [x] Generate and prepare all queries ( 400k )
- [x] Save model weights, with base graph layer separate from cardinality estimation head
- [ ] Use three models to do RL-based finetuning, Tim's model (but bigger), seperate encoding for incoming outgoing, and
  model where predicates are edges with encoding of type of connection. Use both PPO and actor-critic Q-learning 
- [ ] Validate if dropout improves generalization
- [x] IMPORTANT: Separate occurrences / triple pattern sizes calculation from dataset loading, instead make it preprocessing step that is optional depending on if a file location for this information is given.
- [x] IMPORTANT: Annotate the query with the type of query it is.
- [ ] Compare the distribution of cardinalities and predicates of my sampler vs maribel's sampler.
- [ ] IMPORTANT: Generate full dataset of queries
- [ ] Investigate the use of log(cardinality_tp) or log(cardinality_occurrences) in features to scale according to target
- [ ] IMPORTANT: Proper RDF2Vec computation run using own code
- [ ] IMPORTANT: Implement seperate encoding of incoming and outgoing predicates
- [ ] IMPORTANT: Run experiments with different combinations of features to determine best performing pretrain task.
- [ ] Generate queries with less repeating predicates (more diversity in predicates and shapes) and hopefully less massive result sets
- [x] Save different query types separately

- [x] Save predictions made by model in epoch (Save as: [{query, query_type, pred, actual, q-error}, ..]), seperated by train, 
val, and test files. As we have now, save without query 
- [x] Create code for experiment visualization, analysis etc. Violin plots, q-error plots with distribution, epoch development etc.
- [x] Load model weights
- [x] Use single passed around loss function defined in main pretrain function
- [x] Test performance cardinality estimation on completely unseen benchmark queries
- [ ] Test impact of number of queries on unseen benchmark queries
- [x] Find number of parameters in our model
- [x] Use cardinality estimator for join order optimization
- [x] Create system for easy configs for experiment running
- [ ] Generate LUBM benchmark data and test on that
- [ ] Move query generation + execution to this repository and save path, subject star, object star, complex queries separately for more fine-grained analysis.
- [x] Move RDF2Vec to this repository
- [ ] Use pyRDF2Vec on query entities instead of on all entities in graph (for scalability), like in paper recently published
- [ ] Use the samplers from pyRDF2Vec
- [ ] Parallel random walking?

## Ideas:

- Try to use pretraining with deep Q-learning (use actor-critic model)
- Infer RDF2Vec embeddings when new ones arrive by taking random walks starting from this embedding and taking average of the random walk embeddings.
You could even train a model that reconstructs embeddings given random walks, so randomly _corrupt_ embedding, then give it random walks and apply model on random walks so it regenerates the embedding.
- Imitation learning from pretrained cardinality estimator. Reasoning: we copy the cardinality estimator in our weights, but as we get better at imitating we also start including more exploration.
  This exploration might help us find better join plans as we are not bound to possibly inefficient cardinality estimations
  (meaning some accuracy might be redundant, while others might be very necessary).

## Experiments:
- Compare my RDF2Vec vs Tim's RDF2Vec on Tim's model (large / small) (4 experiments)
- Check comparison of performance between Tim's model, Model seperated by incoming outgoing, edge-labeled (same size). 
(6, 3 small 3 large)
After implementing RL-based finetuning
- Validate performance of dropout vs no dropout in learned representations (small amounts)