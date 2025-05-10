We're back to Comunica babyyy. In Comunica we will implement a new explain query-process actor called
`query-process-explain-estimate`. In this actor we will do the whole optimize and execute part. However we will put an
estimation context entry. Then JOIN SAMPLING BABY will fill this with estimated plan costs. As it enumerates the entire
join space we can use this as a reward signal to start with, being the ratio between policy chosen cost and
optimal left-deep cost.