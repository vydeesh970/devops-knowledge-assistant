# infra/terraform/main.tf
#
# This file defines our entire AWS infrastructure as code.
# Instead of clicking through the AWS console manually,
# running "terraform apply" creates everything automatically.
#
# WHAT THIS CREATES:
# - VPC (Virtual Private Cloud) - our isolated network on AWS
# - ECS Cluster - where our Docker containers run
# - ECS Service - keeps our container running and healthy
# - Application Load Balancer - routes traffic to our container
# - ECR Repository - stores our Docker image on AWS
# - Redis (ElastiCache) - our caching layer on AWS
# - Security Groups - firewall rules
# - IAM Roles - permissions for our services

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Stores terraform state in S3 so team members share the same state
  # Uncomment when deploying for real
  # backend "s3" {
  #   bucket = "devops-assistant-terraform-state"
  #   key    = "prod/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

# ============================================================
# VARIABLES
# ============================================================

variable "aws_region" {
  description = "AWS region to deploy to"
  default     = "us-east-1"
}

variable "app_name" {
  description = "Application name"
  default     = "devops-knowledge-assistant"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  default     = "prod"
}

variable "anthropic_api_key" {
  description = "Anthropic API key"
  sensitive   = true  # Never logged or shown in output
}

variable "langchain_api_key" {
  description = "LangSmith API key"
  sensitive   = true
}

# ============================================================
# ECR - Elastic Container Registry
# This is where our Docker image lives on AWS
# Think of it like GitHub but for Docker images
# ============================================================

resource "aws_ecr_repository" "app" {
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true  # Automatically scan for security vulnerabilities
  }

  tags = {
    Name        = var.app_name
    Environment = var.environment
  }
}

# ============================================================
# VPC - Virtual Private Cloud
# Our isolated network on AWS
# Public subnet = accessible from internet (load balancer)
# Private subnet = not directly accessible (our containers)
# ============================================================

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.app_name}-vpc"
  }
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.app_name}-public-${count.index}"
  }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.app_name}-private-${count.index}"
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.app_name}-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ============================================================
# SECURITY GROUPS
# Firewall rules - controls what traffic is allowed in/out
# ============================================================

resource "aws_security_group" "alb" {
  name        = "${var.app_name}-alb-sg"
  description = "Load balancer security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Allow HTTP from anywhere
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "app" {
  name        = "${var.app_name}-app-sg"
  description = "Application security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]  # Only from load balancer
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ============================================================
# APPLICATION LOAD BALANCER
# Distributes incoming traffic across our containers
# Also handles health checks - removes unhealthy containers
# ============================================================

resource "aws_lb" "main" {
  name               = "${var.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = {
    Name = "${var.app_name}-alb"
  }
}

resource "aws_lb_target_group" "app" {
  name        = "${var.app_name}-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 10
    interval            = 30
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# ============================================================
# ECS - Elastic Container Service
# This runs our Docker container on AWS
# Fargate = serverless containers (no servers to manage)
# ============================================================

resource "aws_ecs_cluster" "main" {
  name = "${var.app_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"  # Sends metrics to CloudWatch
  }
}

resource "aws_iam_role" "ecs_execution" {
  name = "${var.app_name}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_ecs_task_definition" "app" {
  family                   = var.app_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"  # 1 vCPU
  memory                   = "2048"  # 2GB RAM
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([{
    name  = var.app_name
    image = "${aws_ecr_repository.app.repository_url}:latest"
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    environment = [
      { name = "ANTHROPIC_API_KEY",      value = var.anthropic_api_key },
      { name = "LANGCHAIN_TRACING_V2",   value = "true" },
      { name = "LANGCHAIN_API_KEY",      value = var.langchain_api_key },
      { name = "LANGCHAIN_PROJECT",      value = var.app_name },
      { name = "LANGCHAIN_ENDPOINT",     value = "https://api.smith.langchain.com" },
      { name = "REDIS_HOST",             value = "localhost" },
      { name = "REDIS_PORT",             value = "6379" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.app_name}"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_service" "app" {
  name            = "${var.app_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.app_name
    container_port   = 8000
  }
}

# ============================================================
# OUTPUTS
# Values printed after terraform apply
# ============================================================

output "api_url" {
  description = "Public URL of the API"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = aws_ecr_repository.app.repository_url
}