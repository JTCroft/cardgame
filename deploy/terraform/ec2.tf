data "aws_ami" "al2023_arm" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-arm64"]
  }
  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

resource "aws_instance" "web" {
  ami                    = data.aws_ami.al2023_arm.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.web.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  # cloud-init only ever runs this on the instance's *first* boot - see the
  # comment at the top of the template for how DNS still gets updated on
  # every later start.
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    hosted_zone_id  = var.hosted_zone_id
    dns_name        = var.subdomain
    github_repo_url = var.github_repo_url
    git_ref         = var.git_ref
  })
  user_data_replace_on_change = true

  tags = {
    Name    = var.project_tag
    Project = var.project_tag
  }
}
