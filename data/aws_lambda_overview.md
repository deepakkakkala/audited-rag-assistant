# AWS Lambda Architecture and Limits

AWS Lambda is a serverless compute service that runs code in response to events and automatically manages the underlying compute resources.

## Execution Environment Lifecycles
When a Lambda function is invoked, the service provisions an execution environment. The initialization phase (Init) includes extension initialization, runtime initialization, and function code initialization. If a function is called again while the container is warm, the initialization phase is skipped.

## Concurrency and Scaling
The default regional concurrency limit for an AWS account is 1,000 concurrent executions across all functions. Lambda allocates burst concurrency between 500 and 3,000 depending on the AWS region, followed by scaling at 500 instances per minute until concurrency limits are reached.

## Memory and Ephemeral Storage
Lambda functions can be configured with memory ranging from 128 MB to 10,240 MB (10 GB) in 1 MB increments. Ephemeral storage allocated to the `/tmp` directory can be configured between 512 MB and 10,240 MB.