data "aws_caller_identity" "current" {}

# --- Role assumed by the EC2 instance itself -------------------------------
# Used at every boot to point Route 53 at whatever public IP that boot got,
# and to allow SSM Session Manager shell access instead of SSH/key pairs.

resource "aws_iam_role" "ec2" {
  name = "${var.project_tag}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = { Project = var.project_tag }
}

resource "aws_iam_role_policy" "ec2_route53" {
  name = "${var.project_tag}-ec2-route53"
  role = aws_iam_role.ec2.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "UpdateOwnDnsRecordOnBoot"
      Effect   = "Allow"
      Action   = "route53:ChangeResourceRecordSets"
      Resource = "arn:aws:route53:::hostedzone/${var.hosted_zone_id}"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project_tag}-ec2-profile"
  role = aws_iam_role.ec2.name
}

# --- Human operator IAM user -----------------------------------------------
# Two policies: broad-ish "provisioning" for infra changes (rarely used,
# only needed when you re-run `terraform apply`), and a narrow "operator"
# policy for routine start/stop + shell access, gated by the Project tag.
#
# No access key is created here on purpose - an IAM access key stored in
# Terraform state is a secret sitting in plaintext on disk. Generate it
# out-of-band once, per the instructions in deploy/DEPLOYMENT.md.

data "aws_iam_policy_document" "provisioning" {
  statement {
    sid    = "EC2Provisioning"
    effect = "Allow"
    actions = [
      "ec2:RunInstances",
      "ec2:TerminateInstances",
      "ec2:CreateTags",
      "ec2:CreateSecurityGroup",
      "ec2:DeleteSecurityGroup",
      "ec2:AuthorizeSecurityGroupIngress",
      "ec2:RevokeSecurityGroupIngress",
      "ec2:CreateVolume",
      "ec2:AttachVolume",
      "ec2:CreateSnapshot",
      "ec2:Describe*",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "InstanceProfileForEC2"
    effect = "Allow"
    actions = [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:CreatePolicy",
      "iam:DeletePolicy",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:CreateInstanceProfile",
      "iam:DeleteInstanceProfile",
      "iam:AddRoleToInstanceProfile",
      "iam:RemoveRoleFromInstanceProfile",
      "iam:GetRole",
      "iam:GetInstanceProfile",
      "iam:GetPolicy",
      "iam:GetPolicyVersion",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:ListInstanceProfilesForRole",
      "iam:TagRole",
      "iam:TagPolicy",
      "iam:TagInstanceProfile",
    ]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project_tag}-*",
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/${var.project_tag}-*",
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:instance-profile/${var.project_tag}-*",
    ]
  }

  # Deliberately scoped: only lets you hand a cardgame-* role to EC2, not
  # to anything else, and not any other role. Without this condition a
  # user with iam:PassRole could attach an unrelated, more privileged role
  # to a resource they control and escalate.
  statement {
    sid       = "PassInstanceRoleToEC2Only"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project_tag}-*"]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ec2.amazonaws.com"]
    }
  }

  statement {
    sid    = "Route53Setup"
    effect = "Allow"
    actions = [
      "route53:ChangeResourceRecordSets",
      "route53:GetHostedZone",
      "route53:ListResourceRecordSets",
    ]
    resources = ["arn:aws:route53:::hostedzone/${var.hosted_zone_id}"]
  }
}

resource "aws_iam_policy" "provisioning" {
  count       = var.create_iam_user ? 1 : 0
  name        = "${var.project_tag}-provisioning"
  description = "One-time/occasional infra changes for the cardgame deployment (terraform apply)."
  policy      = data.aws_iam_policy_document.provisioning.json
}

data "aws_iam_policy_document" "operator" {
  statement {
    sid       = "StartStopOnlyThisInstance"
    effect    = "Allow"
    actions   = ["ec2:StartInstances", "ec2:StopInstances"]
    resources = ["arn:aws:ec2:${var.aws_region}:${data.aws_caller_identity.current.account_id}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/Project"
      values   = [var.project_tag]
    }
  }

  # ec2:Describe* doesn't support resource-level restriction in IAM - it's
  # read-only so the blast radius of leaving it unscoped is low.
  statement {
    sid       = "DescribeIsNotResourceScopable"
    effect    = "Allow"
    actions   = ["ec2:DescribeInstances", "ec2:DescribeInstanceStatus"]
    resources = ["*"]
  }

  statement {
    sid     = "SessionManagerShellAccess"
    effect  = "Allow"
    actions = ["ssm:StartSession", "ssm:TerminateSession", "ssm:ResumeSession"]
    resources = [
      "arn:aws:ec2:${var.aws_region}:${data.aws_caller_identity.current.account_id}:instance/*",
      "arn:aws:ssm:${var.aws_region}::document/SSM-SessionManagerRunShell",
    ]

    condition {
      test     = "StringEquals"
      variable = "ssm:resourceTag/Project"
      values   = [var.project_tag]
    }
  }

  statement {
    sid    = "SsmDescribeIsNotResourceScopable"
    effect = "Allow"
    actions = [
      "ssm:DescribeSessions",
      "ssm:DescribeInstanceInformation",
      "ssm:GetConnectionStatus",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "operator" {
  count       = var.create_iam_user ? 1 : 0
  name        = "${var.project_tag}-operator"
  description = "Day-to-day start/stop + shell access for the cardgame instance."
  policy      = data.aws_iam_policy_document.operator.json
}

resource "aws_iam_user" "operator" {
  count = var.create_iam_user ? 1 : 0
  name  = var.iam_user_name
  tags  = { Project = var.project_tag }
}

resource "aws_iam_user_policy_attachment" "provisioning" {
  count      = var.create_iam_user ? 1 : 0
  user       = aws_iam_user.operator[0].name
  policy_arn = aws_iam_policy.provisioning[0].arn
}

resource "aws_iam_user_policy_attachment" "operator" {
  count      = var.create_iam_user ? 1 : 0
  user       = aws_iam_user.operator[0].name
  policy_arn = aws_iam_policy.operator[0].arn
}
