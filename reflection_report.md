# Reflection Report

I started with 4 agents: an orchestrator, an inventory agent, a quoting agent and a sales agent. As you can see in workflow_diagram.png, the customer talks to the orchestrator, and the orchestrator talks to the inventory agent, the quoting agent and the sales agent, who each have two or three tools to work with. This setup worked relatively well right from the start, though there were a few details to address. 

First, I noticed that something weird was going on with the dates. The model, gpt-4o-mini, defaulted to its ~2023 training cutoff. This was leaking incorrect dates in the communication. I fixed it by injecting an authoritative date and overriding any hallucination at the tool level.

Second, there was a bug where any unknown query containing the word "paper" silently mis-mapped to "A4 paper" at $0.05 because of greedy token-overlap matching. I fixed it in resolve_item_name, e.g. by detecting ambiguous ties so unknown items are declined instead of mis-priced.

Third, there was quite some computer syntax being communicated to the customer. This was fixed by adding a deterministic clean_customer_response post-processor that strips the scaffolding and keeps the detailed body (with any non-trivial additional context) so customers see plain, professional language instead of tool internals.

The file test_results.csv shows the latest run, after all bugs were resolved. Across the 20 sample requests, the system changed the cash balance on 11 of them and successfully finalized 6 sales, while the remaining requests were declined with a clear reason (out of stock, item not in catalog, or insufficient inventory). A particular strength is transparency: every response states the item, quantity, price, and fulfillment status in plain language, with no internal tooling, margins, or hallucinated dates leaking through. The system also correctly declines ambiguous or unknown items rather than mis-pricing them.

On top of this, to prepare for the pixel art animation (see below), I introduced a customer agent. To avoid messing with the standard solution, I put this one in customer_negotiation_demo.py.

My colleagues at work recently told me about a GitHub repository that allowed making a pixel art animation of agent interactions. I wanted to use it for this project, but couldn't, because it was built for Claude Code agents. So instead, together with GitHub Copilot, I built my own version. How it works is explained in the README. Please check it out, as I put quite some work into it. It is a zero-dependency, read-only viewer (Python stdlib http.server) that renders a pixel-art office on port 8000 to visualize live agent activity. To get this css right in terms of how to see the agents versus their desk versus the monitor and the bubbles with the text took quite a number of iterations (see also the commit history).

Further work could be centered around the following: there is some efficiency work to do on the restocking policy, where items could be re-stocked before executing a customer order, we had an incident where a maximum number of steps was taken before the customer was satisfied, and there is a weird economic model where the unit price is used for both the restock cost and the sales price, leaving zero per-unit margin. Discounts are direct losses, which is not too good.

It was a lot of fun to make this application!
