mkdir docker

%%writefile docker/dockerfile

FROM python:3.7-slim-buster
RUN pip3 install pandas scikit-learn
ENV PYTHONUNBUFFERED=TRUE
COPY processing.py .

!docker build -t $ecr_repository docker
!aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com
!aws ecr create-repository --repository-name $ecr_repository
!docker tag {ecr_repository + tag} $processing_repository_uri
!docker push $processing_repository_uri
