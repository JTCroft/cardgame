resource "aws_route53_record" "web" {
  zone_id = var.hosted_zone_id
  name    = var.subdomain
  type    = "A"
  ttl     = 60
  records = [aws_instance.web.public_ip]

  # After the first apply, cardgame-dns-update.service (running on the
  # instance) keeps this record's value current on every boot, out of
  # band from Terraform - don't fight it back to a stale IP on later
  # applies.
  lifecycle {
    ignore_changes = [records]
  }
}
