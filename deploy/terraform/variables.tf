variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_tag" {
  description = "Value used for the Project tag on every resource, and as the name prefix for IAM/EC2 resources. The operator IAM policy's start/stop permission is gated on this tag."
  type        = string
  default     = "cardgame"
}

variable "hosted_zone_id" {
  description = "Route 53 hosted zone ID that already holds your domain (from the zone you procured)."
  type        = string
}

variable "subdomain" {
  description = "Full DNS name the game will be served on, e.g. play.example.com."
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type. t4g.* is Graviton/arm64 - cheaper than the t3.* x86 equivalent."
  type        = string
  default     = "t4g.micro"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. This is the one cost that's charged whether the instance is running or stopped. The AL2023 arm64 AMI's snapshot currently requires at least 30."
  type        = number
  default     = 30
}

variable "github_repo_url" {
  description = "Git URL the instance clones the app from at first boot."
  type        = string
  default     = "https://github.com/JTCroft/cardgame.git"
}

variable "git_ref" {
  description = "Branch/tag to deploy."
  type        = string
  default     = "main"
}

variable "create_iam_user" {
  description = "Whether Terraform should create the human operator IAM user and its two policies. Set false if you'd rather manage that user by hand."
  type        = bool
  default     = true
}

variable "iam_user_name" {
  description = "Name of the IAM user created for day-to-day start/stop and initial provisioning."
  type        = string
  default     = "cardgame-operator"
}
