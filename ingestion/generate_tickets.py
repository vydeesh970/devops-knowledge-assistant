"""
generate_tickets.py

What this file does:
Generates 300 realistic DevOps support tickets and saves them as a CSV file.
These tickets simulate what a real company's internal ticketing system looks like
(think Jira or ServiceNow). This becomes the STRUCTURED data source in our project
that the AI agent queries for things like:
"Show me all open P1 Kubernetes tickets"
"""

import csv
import random
from datetime import datetime, timedelta

# Setting a seed means we get the same "random" data every time we run this
# This is important for reproducibility - a good engineering practice
random.seed(42)

# These are realistic DevOps problems that engineers actually face
CATEGORIES = {
    "kubernetes": [
        "Pod stuck in CrashLoopBackOff after deployment",
        "ImagePullBackOff error on new container image",
        "Service not routing traffic to pods",
        "Persistent volume claim stuck in Pending state",
        "Node showing NotReady status intermittently",
        "Horizontal Pod Autoscaler not scaling under load",
        "ConfigMap changes not reflected in running pods",
        "Ingress returning 502 Bad Gateway",
        "OOMKilled errors on production workload",
        "RBAC permission denied for service account",
        "DNS resolution failing inside cluster",
        "Rolling update stuck, old pods not terminating",
    ],
    "fastapi": [
        "Dependency injection raising 422 on valid request",
        "Async endpoint timing out under concurrent load",
        "CORS errors blocking frontend requests",
        "Pydantic validation error on nested models",
        "Background task not executing after response sent",
        "OpenAPI docs not rendering for new router",
        "Authentication middleware not applied to subroute",
        "WebSocket connection dropping after 60 seconds",
        "File upload endpoint failing for large files",
        "Database session not closing, connection pool exhausted",
    ],
    "infrastructure": [
        "Terraform apply failing on state lock",
        "CI/CD pipeline failing at Docker build stage",
        "SSL certificate expired on production endpoint",
        "Redis connection refused after restart",
        "S3 bucket access denied from EC2 instance",
        "High latency reported on API gateway",
    ],
}

# P1 = most critical (production down), P4 = low priority
PRIORITIES = ["P1", "P2", "P3", "P4"]

# Most tickets are P3 - not everything is a crisis
PRIORITY_WEIGHTS = [0.1, 0.25, 0.4, 0.25]

STATUSES = ["open", "in_progress", "resolved", "closed"]

ASSIGNEES = ["a.patel", "j.kim", "m.santos", "r.oconnor", "unassigned"]


def generate_tickets(n=300):
    """Generate n realistic support tickets"""
    tickets = []
    start_date = datetime(2025, 1, 1)

    for i in range(1, n + 1):
        # Pick a random category and a random problem from that category
        category = random.choice(list(CATEGORIES.keys()))
        title = random.choice(CATEGORIES[category])

        # Pick priority based on weights (P3 most common, P1 rarest)
        priority = random.choices(PRIORITIES, weights=PRIORITY_WEIGHTS)[0]

        # P1 tickets get resolved faster (realistic - teams prioritize these)
        if priority == "P1":
            status = random.choices(
                STATUSES,
                weights=[0.1, 0.2, 0.4, 0.3]
            )[0]
        else:
            status = random.choices(
                STATUSES,
                weights=[0.3, 0.2, 0.25, 0.25]
            )[0]

        # Spread tickets across the past year
        created = start_date + timedelta(
            days=random.randint(0, 530),
            hours=random.randint(0, 23)
        )

        # Resolved tickets have a resolution time
        resolved = None
        if status in ("resolved", "closed"):
            resolved = created + timedelta(hours=random.randint(1, 240))

        tickets.append({
            "ticket_id": f"OPS-{1000 + i}",
            "title": title,
            "category": category,
            "priority": priority,
            "status": status,
            "assignee": "unassigned" if status == "open"
                        else random.choice(ASSIGNEES[:-1]),
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "resolved_at": resolved.strftime("%Y-%m-%d %H:%M:%S")
                           if resolved else "",
            "description": (
                f"{title}. Reported in production environment, "
                f"affecting {random.choice(['1 service', 'multiple services', 'a subset of users', 'all users in us-east-1'])}."
            ),
        })

    return tickets


def main():
    tickets = generate_tickets(300)

    # Save to CSV file in our data folder
    out_path = "data/raw/tickets.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tickets[0].keys())
        writer.writeheader()
        writer.writerows(tickets)

    print(f"✅ Generated {len(tickets)} tickets → {out_path}")

    # Show a breakdown so we can verify the data looks realistic
    by_status = {}
    by_priority = {}
    for t in tickets:
        by_status[t["status"]] = by_status.get(t["status"], 0) + 1
        by_priority[t["priority"]] = by_priority.get(t["priority"], 0) + 1

    print(f"📊 Status breakdown: {by_status}")
    print(f"📊 Priority breakdown: {by_priority}")


if __name__ == "__main__":
    main()