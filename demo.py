"""Full demo of all Aegis components — environment + attacks."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aegis.environment import (
    WorkspaceSimulator,
    IntentAuditor,
    MemoryMonitor,
    RewardCalculator,
)
from aegis.attacks import (
    DirectExfilAttack,
    ConfusedDeputyAttack,
    MemoryPoisonAttack,
    SupplyChainAttack,
)

SEP = "=" * 60
THIN = "-" * 60


def demo_attacks():
    print(f"\n{SEP}")
    print("  AEGIS — Attack Scripts")
    print(SEP)

    attacks = [
        DirectExfilAttack(),
        ConfusedDeputyAttack(),
        MemoryPoisonAttack(),
        SupplyChainAttack(),
    ]

    for atk in attacks:
        print(f"\n{atk!r}")
        print(f"  Steps: {atk.total_steps}  "
              f"Malicious: {atk.malicious_step_count}  "
              f"Benign: {atk.benign_step_count}")
        print()

        while not atk.is_complete():
            action = atk.get_next_action()
            idx = action["step_index"]
            mal = "MAL" if action["is_malicious"] else "   "
            cmd = action["command"]
            intent = action["stated_intent"]
            print(f"  [{idx}] [{mal}] $ {cmd}")
            print(f"         intent: {intent!r}")

        # Grade: perfect sentinel
        perfect = [
            "block" if s["is_malicious"] else "allow"
            for s in atk.get_full_script()
        ]
        grade = atk.grade(perfect)
        print(f"  Perfect score: {grade['score']}")

        # Grade: naive allow-all sentinel
        naive = ["allow"] * atk.total_steps
        grade2 = atk.grade(naive)
        print(f"  Naive (allow-all) score: {grade2['score']}")

        # Reset check
        atk.reset()
        assert atk.current_step == 0
        assert not atk.is_complete()
        print("  Reset OK")


def demo_full_episode():
    print(f"\n{SEP}")
    print("  AEGIS — Simulated Episode (Easy Attack)")
    print(SEP)

    ws = WorkspaceSimulator()
    auditor = IntentAuditor()
    memory = MemoryMonitor()
    rewards = RewardCalculator()
    attack = DirectExfilAttack()

    # Baseline memory
    baseline = "You are a helpful coding assistant."
    memory.take_snapshot(baseline)

    sentinel_actions = []

    while not attack.is_complete():
        action = attack.get_next_action()
        idx = action["step_index"]
        cmd = action["command"]
        intent = action["stated_intent"]
        is_mal = action["is_malicious"]

        # Auditor scores intent vs command
        match_score = auditor.audit(intent, cmd)
        threat = auditor.classify_threat(cmd)

        # Simple rule-based sentinel: block if match < 0.3 or threat != clean
        if match_score < 0.3 or threat["threat_type"] != "clean":
            sentinel_decision = "block"
        else:
            sentinel_decision = "allow"

        sentinel_actions.append(sentinel_decision)

        # Was the decision correct?
        correct = (sentinel_decision == "block") == is_mal

        # Execute in workspace (only if allowed)
        if sentinel_decision == "allow":
            ws_result = ws.execute_command(cmd)
            ht_triggered = ws_result.get("honeytoken_triggered", False)
            leaked = ws_result.get("exfil_attempt", False)
        else:
            ht_triggered = False
            leaked = False

        # Reward
        drift = memory.check_drift(baseline)
        step_r = rewards.calculate_step_reward(
            action_type=sentinel_decision,
            intent_match_score=match_score,
            honeytoken_triggered=ht_triggered,
            threat_detected=threat["threat_type"] != "clean",
            agent_decision_correct=correct,
            secrets_leaked=leaked,
            memory_drift=drift,
            sandboxed=False,
        )

        status = "BLOCKED" if sentinel_decision == "block" else "ALLOWED"
        correct_mark = "OK" if correct else "WRONG"
        print(f"  [{idx}] $ {cmd}")
        print(f"       intent={intent!r}  match={match_score:.2f}")
        print(f"       threat={threat['threat_type']:<20s} "
              f"sentinel={status:<8s} [{correct_mark}]  reward={step_r:+.3f}")

    # Final grade
    grade = attack.grade(sentinel_actions)
    print(f"\n  Episode Score:  {grade['score']}")
    print(f"  Blocks Correct: {grade['blocks_correct']}  "
          f"Allows Correct: {grade['allows_correct']}  "
          f"False Negatives: {grade['false_negatives']}  "
          f"False Positives: {grade['blocks_incorrect']}")
    print(f"  Cumulative Reward: {rewards.cumulative_reward:+.3f}  "
          f"Average: {rewards.average_reward:+.3f}")


def main():
    demo_attacks()
    demo_full_episode()
    print(f"\n{SEP}")
    print("  All systems operational!")
    print(SEP)


if __name__ == "__main__":
    main()
