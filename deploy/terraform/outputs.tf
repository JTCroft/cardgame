output "instance_id" {
  description = "Use this with `aws ec2 start-instances` / `stop-instances`."
  value       = aws_instance.web.id
}

output "public_ip" {
  description = "Current public IP - only meaningful while the instance is running, and only until the next stop/start."
  value       = aws_instance.web.public_ip
}

output "url" {
  value = "https://${var.subdomain}"
}

output "operator_iam_user" {
  description = "Run `aws iam create-access-key --user-name <this>` once to get credentials - not created by Terraform so the secret never lands in state."
  value       = var.create_iam_user ? aws_iam_user.operator[0].name : null
}
