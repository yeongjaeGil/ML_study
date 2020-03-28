# Spark 
---
is a big data solution that has been proven to be easier and faster than Hadoop MapReduce. The main difference between Spark and MapReduce is that Spark runs computations in memory during the later on the hard disk. It allows high-speed access and data processing, reducing times from hours to minutes.
---
#### 스파크 역할
- computational engine, meaning it takes care of the scheduling, distributing and monitoring applications. Each task is done across various worker machines called computing cluster. A computer cluster refers to the division of tasks. In the end, all the tasks are aggregated to produce an output. The Spark admin gives a 360 overview of various Spark Jobs.
---
WHY use Spark?
- original process:
    - Load the data to the disk
    - Process / analyze the data
    - Build the machine learning model
    - Store the prediction back to disk
- The problem arises if the data scientist wants to process data that's too big for one computer.
    - 전체 데이터가 아니라 좋은 샘플링을 통해 모델을 만드는 작업을 해왔다.
        - 여기서 생기는 문제점
            - is the dataset reflecting the real world?
            - Does the data include a specific example?
            - Is the model fit for sampling?
- solution
    - split the problem up onto multiple computers.
        - Parallel computing comes with multiple problems as well.
    - Pyspark gives the data scientist an API that can be used to solve the parallel 
---


Q) 결국 어떨 때 pyspark를 사용하는 것이 효율적인지를 명확히 짚을 수 있어야 한다.