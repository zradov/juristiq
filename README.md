![Project Status](https://img.shields.io/badge/status-active-brightgreen)
## *This is a work in progress*

## 1. Project description
The project implements an autonomous legal research and drafting agent leveraging AWS-native AI services.
The system can:
* research legal contracts;
* extract and rank relevant clauses and precedents;
* draft structured memos, summaries, or contract redlines;
* perform automated contract review for assignment restrictions, termination clauses, indemnities, and more;
* operate in autonomous or human-in-the-loop mode with escalation for ambiguous cases.

**Key technologies**: 
  * Amazon Bedrock (Nova, AgentCore);
  * Amazon SageMaker, OpenSearch with vector search, Amazon Q, and the CUAD dataset for fine-tuning.

## 2. Project structure 

## 3. Architecture

The project integrates multiple AWS services:
* **Amazon S3** - raw contracts, CUAD dataset, model artifacts;
* **Amazon Textract** - extract text from PDFs;
* Amazon SageMaker - fine-tuning & hosting embedding model
* **Amazon Bedrock Knowledge Bases** - supports an end-to-end RAG workflow;
* **OpenSearch (vector)** - custom search logic or hybrid retrieval, custom indexing, ranking and filtering;
* **Amazon Bedrock AgentCore** - deploying and scaling dynamic AI agents and tools;
*  **Amazon Bedrock Nova** - reasoning LLM for drafting/explanations;
*  **Amazon Q** - front-end for users interaction;
* **AWS Step Functions**  optional workflow including human-in-the-loop review.

<br/>![Figure 2.](./arch/juristiq.png)

## 4. Agentic workflow

The workflow includes following steps:

1. **Ingestion**: Contracts uploaded to S3 → Textract → chunked into clauses.
2. **Indexing**: Clauses embedded via SageMaker endpoint → vectors stored in OpenSearch.
3. **Query**: User asks a legal question via Amazon Q (or API).
4. **Retrieval**: AgentCore retrieves top-k relevant clauses.
5. **Reasoning**: Nova LLM synthesizes answer, cites sources, drafts clauses or memos.
6. **Validation**: Confidence thresholds decide auto-response vs escalation to human review.
7. **Storage**: Contract saved in S3, optionally contract's metadata could be stored in Amazon Bedrock Knowledge Bases.

## 5. The training process

## 6. Data annotations

## 7. Prerequisities

## 8. Infrastructure as code

## 9. Running the app

## 10. Running the unit tests

