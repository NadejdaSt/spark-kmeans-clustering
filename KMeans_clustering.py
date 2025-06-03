from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.appName("OptimalKMeans").getOrCreate()

# File path for input data
input_file = "Clustering_dataset.txt"

# Load data
data = spark.read.option("delimiter", " ").csv(input_file, inferSchema=True)
data = data.withColumnRenamed("_c0", "x").withColumnRenamed("_c1", "y")

# Prepare data for KMeans
assembler = VectorAssembler(inputCols=["x", "y"], outputCol="features")
assembled_data = assembler.transform(data)

# Determine optimal K using silhouette score
low = int(input("Input minimum K: "))
high = int(input("Input maximum K: "))
k_values = range(low, high)
silhouette_scores = []

print(f"Calculating the most optimal K in range ({low}, {high})")
for k in k_values:
    kmeans = KMeans(featuresCol="features", k=k, seed=42)
    model = kmeans.fit(assembled_data)
    predictions = model.transform(assembled_data)

    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette)
    print(f"K={k}, Silhouette Score={silhouette:.4f}")

# Find the best K
optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"Optimal K: {optimal_k}")

# Fit model with optimal K
optimal_kmeans = KMeans(featuresCol="features", k=optimal_k, seed=42)
optimal_model = optimal_kmeans.fit(assembled_data)
final_predictions = optimal_model.transform(assembled_data)

# Collect data for visualization
cluster_centers = optimal_model.clusterCenters()
clusters = final_predictions.select("x", "y", "prediction").collect()

# Convert to lists for plotting
x_vals = [row["x"] for row in clusters]
y_vals = [row["y"] for row in clusters]
predictions = [row["prediction"] for row in clusters]

# Plot clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x_vals, y_vals, c=predictions, cmap="viridis", alpha=0.6, edgecolors="w")
plt.scatter([center[0] for center in cluster_centers], [center[1] for center in cluster_centers], s=200, c="red", label="Centers")
plt.title(f"Clusters for Optimal K={optimal_k}")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.colorbar(scatter, label="Cluster")
plt.show()

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker="o", linestyle="--", color="b")
plt.title("Silhouette Scores for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Stop Spark session
spark.stop()
