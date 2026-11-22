# SARplus

Simple Algorithm for Recommendation (SAR) is a neighborhood based
algorithm for personalized recommendations based on user transaction
history. SAR recommends items that are most **similar** to the ones
that the user already has an existing **affinity** for. Two items are
**similar** if the users that interacted with one item are also likely
to have interacted with the other. A user has an **affinity** to an
item if they have interacted with it in the past.

SARplus uses PySpark 3.x and a matching Scala 2.12 SARPlus JAR. Install the
Python package on Linux, then supply the JAR for your Spark version when
starting the Spark session. The wheel contains the native prediction module;
Spark loads the data source from the JAR.
More details can be found at
[sarplus@microsoft/recommenders](https://github.com/microsoft/recommenders/tree/main/contrib/sarplus).
