import mlflow
import os
import pandas as pd
from dotenv import load_dotenv
import time
from openai import OpenAI
from transformers import pipeline
#import dagshub


load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype="auto",
    device_map="auto"
)

#mlflow.set_tracking_uri("https://github.com/radhika-singh-10/ML-FLOW-Experiments/tree/dev-genai")
eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)

mlflow.set_experiment("LLM Evaluation - Mistral HF")

# Prompt template
prompt_prefix = "Answer the following question in two sentences:\n\nQ: "


def query_mistral(question):
    prompt = f"[INST] Answer the following question in two sentences:\n{question} [/INST]"
    start = time.time()
    output = llm(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
    latency = time.time() - start
    answer = output.split("[/INST]")[-1].strip()
    return answer, latency


with mlflow.start_run():
    predictions = []
    latencies = []

    for question in eval_data["inputs"]:
        answer, latency = query_mistral(question)
        predictions.append(answer)
        latencies.append(latency)

    eval_data["predicted_answer"] = predictions
    eval_data["latency_seconds"] = latencies

    eval_data.to_csv("mistral_eval.csv", index=False)
    mlflow.log_artifact("mistral_eval.csv")
    mlflow.log_metric("avg_latency", sum(latencies) / len(latencies))
    mlflow.log_param("model", "mistralai/Mistral-7B-Instruct-v0.1")

    print(eval_data[["inputs", "predicted_answer", "latency_seconds"]])