import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql import SparkSession
# Added 'lower', 'when', 'col', 'lit', 'isnan' for new transformations and quality checks
from pyspark.sql.functions import from_json, col, current_timestamp, to_date, year, month, dayofmonth, lower, when, lit, isnan
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

# --- Glue Job Boilerplate (Essential for running on AWS Glue) ---
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# --- SCHEMA DEFINITION ---
schema = StructType([
    StructField("event_id", StringType(), True),         # Unique identifier for each event
    StructField("timestamp", StringType(), True),        # Timestamp of the event as a string (will be cast later)
    StructField("user_id", StringType(), True),          # Identifier for the user
    StructField("event_type", StringType(), True),       # Type of event (e.g., "browse_product", "add_to_cart", "purchase")
    StructField("product_id", StringType(), True),       # Identifier for the product
    StructField("category", StringType(), True),         # Category of the product
    StructField("price", DoubleType(), True),            # Price of the item (can be null for non-purchase events)
    StructField("quantity", IntegerType(), True)         # Quantity of items (can be null for non-cart/purchase events)
])

# --- S3 BUCKET AND REGION CONFIGURATION ---
RAW_S3_BUCKET_NAME = "ecom-analytics-raw-data-alankrit-123"
PROCESSED_S3_BUCKET_NAME = "ecom-analytics-processed-data-alankrit-123"
AWS_REGION = "ap-south-1"

# --- Explicit S3A and Committer Configurations ---
spark.conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark.conf.set("spark.sql.sources.commitProtocolClass", "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol")
spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
spark.conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")


# --- READ RAW DATA FROM S3 USING GLUE NATIVE METHOD ---
raw_s3_path = f"s3://{RAW_S3_BUCKET_NAME}/raw-events/"

raw_dynamic_frame = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={
        "paths": [raw_s3_path],
        "recurse": True
    },
    format="json",
    format_options={
        "multiline": True
    },
    transformation_ctx="raw_input_data"
)

raw_df = raw_dynamic_frame.toDF()

# --- ADVANCED CLEANING AND TRANSFORMATION ---
# 1. Handle Missing Values: Fill null 'price' and 'quantity' with 0
#    Standardize event_type and category to lowercase
cleaned_df = raw_df.select(
    col("event_id"),
    col("timestamp").cast(TimestampType()).alias("event_timestamp"),
    col("user_id"),
    lower(col("event_type")).alias("event_type"),
    col("product_id"),
    lower(col("category")).alias("category"),
    when(col("price").isNull(), lit(0.0)).otherwise(col("price")).alias("price"),
    when(col("quantity").isNull(), lit(0)).otherwise(col("quantity")).alias("quantity"),
    current_timestamp().alias("processing_timestamp")
)

# 2. Derive New Features: Calculate 'total_price' for purchase events
transformed_df = cleaned_df.withColumn(
    "total_price",
    when(col("event_type") == "purchase", col("price") * col("quantity")).otherwise(lit(None))
)

# --- DATA QUALITY CHECKS (NEW SECTION) ---

# Define data quality rules
# Rule 1: Completeness - event_id, user_id, event_type, event_timestamp must not be null
# Rule 2: Validity - price must be >= 0, quantity must be >= 0
valid_records_df = transformed_df.filter(
    (col("event_id").isNotNull()) &
    (col("user_id").isNotNull()) &
    (col("event_type").isNotNull()) &
    (col("event_timestamp").isNotNull()) &
    (col("price") >= 0) & # After null handling, price is not null, so just check range
    (col("quantity") >= 0) # After null handling, quantity is not null, so just check range
)

# Optional: Identify and log invalid records
invalid_records_df = transformed_df.filter(
    (col("event_id").isNull()) |
    (col("user_id").isNull()) |
    (col("event_type").isNull()) |
    (col("event_timestamp").isNull()) |
    (col("price") < 0) |
    (col("quantity") < 0)
)

# Count and log the number of valid and invalid records
total_records = transformed_df.count()
valid_records_count = valid_records_df.count()
invalid_records_count = invalid_records_df.count()

print(f"Data Quality Check Results:")
print(f"  Total records read: {total_records}")
print(f"  Valid records passing quality checks: {valid_records_count}")
print(f"  Invalid records identified: {invalid_records_count}")

if invalid_records_count > 0:
    print(f"WARNING: {invalid_records_count} invalid records were found and will be excluded from the output.")
    # You might also write invalid_records_df to a 'dead letter' S3 path for later investigation
    # For now, we'll just log the warning.

# --- Prepare DataFrame for writing (using only valid records) ---
processed_df_for_write = valid_records_df.withColumn("year", year(col("event_timestamp"))) \
                                        .withColumn("month", month(col("event_timestamp"))) \
                                        .withColumn("day", dayofmonth(col("event_timestamp")))


# --- Diagnostic Print: Check record count before writing (will now reflect valid records only) ---
print(f"Number of records to write after quality checks: {processed_df_for_write.count()}")

# --- WRITE PROCESSED DATA TO A NEW S3 BUCKET IN PARQUET FORMAT ---
output_base_path = f"s3://{PROCESSED_S3_BUCKET_NAME}/processed-ecommerce-data/"

processed_df_for_write.write \
    .format("parquet") \
    .partitionBy("event_type", "year", "month", "day") \
    .mode("append") \
    .save(output_base_path)

# --- Commit the job (Important for AWS Glue) ---
job.commit()

# --- Logging for Confirmation ---
print(f"Successfully processed data from {raw_s3_path} and written to {output_base_path}")