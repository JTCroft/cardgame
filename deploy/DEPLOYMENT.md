# Deploying the web UI to AWS

This deploys `cardgame-web` to a single EC2 instance you start when you want
to play and stop when you don't. See the architecture notes at the bottom
for *why* it's shaped this way. Terraform config lives in
[`deploy/terraform/`](terraform/).

## Prerequisites

- Terraform >= 1.5 (`terraform version`)
- AWS CLI v2, configured with **some** credentials that can create IAM
  roles/policies/users and EC2 resources (your account root, an existing
  admin IAM user, or IAM Identity Center admin permission set). This is a
  one-time bootstrapping identity — after the first `apply` you'll switch
  to the narrower `cardgame-operator` user this creates for everything else.
- A Route 53 hosted zone already covering your domain (you said this is
  procured — grab its zone ID: `aws route53 list-hosted-zones-by-name`).
- This repo's code is pushed to a public (or otherwise fetchable) git
  remote — the instance clones it at boot. Defaults to
  `https://github.com/JTCroft/cardgame.git`, override via `github_repo_url`
  / `git_ref` if you deploy from a fork or branch.

## First-time setup

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: hosted_zone_id, subdomain, aws_region

terraform init
terraform validate
terraform plan    # review what it'll create
terraform apply
```

This creates:
- An EC2 instance role + instance profile (Route 53 update permission,
  SSM Session Manager access)
- A security group (80/443 open, no SSH port — see below)
- The EC2 instance itself, with a boot script that installs Caddy, `uv`,
  clones the repo, and starts `cardgame-web` behind Caddy as systemd services
- A Route 53 A record for `subdomain`, pointed at the instance
- An IAM user (`cardgame-operator` by default) with two policies attached:
  `cardgame-provisioning` (broad-ish, for future `terraform apply` runs) and
  `cardgame-operator` (narrow: start/stop this one tagged instance + SSM
  shell access)

Terraform deliberately does **not** create an access key for that IAM user
— an access key sitting in Terraform state is a plaintext secret on disk.
Create one yourself, once:

```bash
aws iam create-access-key --user-name cardgame-operator
```

Save the output somewhere safe (a password manager, not a repo). Then set
up a local profile so day-to-day commands don't touch your admin
credentials:

```bash
aws configure --profile cardgame
# paste the access key/secret from above
```

From here on, use `--profile cardgame` (or `export AWS_PROFILE=cardgame`)
for start/stop/SSM commands. Keep using your admin credentials only for
`terraform apply` when you change the infra itself.

Wait ~1-2 minutes after `apply` finishes for the boot script to install
everything, issue a TLS cert, and come up. Then check `terraform output
url` in a browser.

## Day-to-day: starting and stopping

```bash
INSTANCE_ID=$(cd deploy/terraform && terraform output -raw instance_id)

# start a session
aws ec2 start-instances --instance-ids "$INSTANCE_ID" --profile cardgame
# wait ~30-60s for boot + the DNS record to update (60s TTL), then:
#   https://<your subdomain> is live

# done playing
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --profile cardgame
```

Worth aliasing both. Game state is in-memory only (see `app.py`'s module
docstring) — stopping mid-game just ends it, there's nothing to lose by
design.

## Shell access (no SSH keys / port 22 needed)

```bash
aws ssm start-session --target "$INSTANCE_ID" --profile cardgame
```

Useful for checking logs:

```bash
sudo systemctl status cardgame-web caddy cardgame-dns-update
sudo journalctl -u cardgame-web -f
```

## Deploying app code changes

The boot script only clones the repo on the instance's *first* boot. To
pick up new commits on an existing instance:

```bash
aws ssm start-session --target "$INSTANCE_ID" --profile cardgame
# once connected:
cd /opt/cardgame
sudo git fetch origin && sudo git reset --hard origin/main
sudo /usr/local/bin/uv sync --extra web
sudo systemctl restart cardgame-web
```

If you'd rather always get a fully clean instance (e.g. after bigger infra
changes), `terraform taint aws_instance.web && terraform apply` replaces it
outright and reruns the boot script from scratch.

## Cost

With occasional use (a few hours a month):

| Item | Cost |
|---|---|
| EC2 t4g.micro, only while running | ~$0.008/hr → pennies/month |
| EBS gp3 8GB root volume (billed even while stopped) | ~$0.65/month |
| Public IPv4 (only while running) | ~$0.005/hr → pennies/month |
| Route 53 hosted zone | $0.50/month |
| **Total** | **~$1-2/month** |

Leaving it running by accident costs at most ~$6-7/month (24/7 t4g.micro)
— not runaway. There's no idle-shutdown timer in this setup (you chose
manual start/stop); see below if you want one added later.

## Tearing down

```bash
cd deploy/terraform
terraform destroy
```

Removes the instance, security group, IAM role/user/policies, and the
Route 53 record. Nothing else in your AWS account is touched.

## Architecture notes

- **Why one instance, not autoscaled/serverless**: `rooms.py` keeps all
  game state in memory, per-process, with no shared store (Redis etc.) —
  see the module docstrings. That rules out anything that could run more
  than one instance at a time, which conveniently is also the cheapest
  option here.
- **Why no Elastic IP**: since Feb 2024 AWS charges $0.005/hr for *every*
  public IPv4, attached or not — a static IP no longer saves money, it just
  costs the same all the time. Instead, EC2 assigns a fresh public IP only
  while running (free the rest of the time), and `cardgame-dns-update.service`
  re-points the Route 53 record at it on every boot.
- **Why not Lightsail**: stopping a Lightsail instance does not stop
  billing — you pay the full monthly plan rate whether it's on or off,
  which defeats "spin up/down on demand."
  [Amazon Lightsail Pricing: 2026 Guide to True Total Cost](https://cloudburn.io/blog/amazon-lightsail-pricing)
- **Why no ALB/Fargate**: an ALB is a fixed ~$16-20/month regardless of
  usage, dwarfing everything else for a low-traffic hobby app that doesn't
  need horizontal scaling anyway.
- **Why no SSH**: the instance role has `AmazonSSMManagedInstanceCore`
  attached, so `aws ssm start-session` gives a shell with no open port 22,
  no key pair to lose, and access controlled entirely through IAM.
