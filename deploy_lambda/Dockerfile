# Use the official Amazon Linux image for Lambda
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory in the container
WORKDIR /deploy_lambda

COPY requirements.txt /deploy_lambda/

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the CMD to your handler (adjust accordingly)
CMD ["dep_lambda.lambda_handler"]

