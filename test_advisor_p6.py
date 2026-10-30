from fractions import Fraction as F

import advisor
import social_media_gt as gt


def test_tw7_lambda_is_fraction_of_baseline_peer_pressure_removed():
    lam = F(1, 3)
    expected = gt.tw7_pressure_reduction(lam=lam, eta=F(0))
    evidence = advisor.run_one_scenario("tw7", {"lam": lam, "eta": F(0)})

    assert F(evidence["results"]["lambda_effective"]) == lam
    assert evidence["results"]["lambda_effective"] == gt.show(
        expected["results"]["lambda_effective"])
    assert evidence["results"]["limit"] == gt.show(expected["results"]["limit"])


def test_tw7_eta_reduces_specified_fraction_by_exact_one_minus_eta():
    lam, eta = F(1, 5), F(1, 2)
    expected = gt.tw7_pressure_reduction(lam=lam, eta=eta)
    evidence = advisor.run_one_scenario("tw7", {"lam": lam, "eta": eta})

    effective = lam * (1 - eta)
    assert F(evidence["results"]["lambda_effective"]) == effective
    assert evidence["results"]["lambda_effective"] == gt.show(
        expected["results"]["lambda_effective"])
    assert evidence["results"]["limit"] == gt.show(expected["results"]["limit"])


def test_tw7_lambda_is_clamped_to_fractional_reduction_domain():
    normalized, notes = advisor.normalize_params("tw7", {"lam": F(3, 2)})
    assert normalized["lam"] == 1
    assert any("fraction must stay in [1/1000, 1]" in note for note in notes)


def test_tw7_strategy_rows_compare_attenuated_and_unattenuated_exact_results():
    lam, eta = F(1, 5), F(1, 2)
    evidence = advisor.run_one_scenario("tw7", {"lam": lam, "eta": eta})
    attenuated = gt.tw7_pressure_reduction(lam=lam, eta=eta)["results"]
    baseline = gt.tw7_pressure_reduction(lam=lam, eta=F(0))["results"]

    assert [row["values"] for row in evidence["strategies"]] == [
        {"effective fraction removed": gt.show(attenuated["lambda_effective"]),
         "adoption limit": gt.show(attenuated["limit"])},
        {"effective fraction removed": gt.show(baseline["lambda_effective"]),
         "adoption limit": gt.show(baseline["limit"])},
    ]


def test_tw7_conclusion_and_digest_explain_lambda_semantics():
    evidence = advisor.run_one_scenario("tw7", {"lam": F(1, 5), "eta": F(1, 2)})
    conclusion = " ".join(evidence["conclusion"]).casefold()
    assert "fraction of baseline peer pressure removed" in conclusion
    assert "lambda * (1 - eta)" in conclusion
    assert "100 * (1 - lambda) * k" in conclusion
    assert "pressure slope" not in conclusion
    assert "neither group coverage nor evasion" in conclusion
    assert F(evidence["results"]["lambda_effective"]) == F(1, 10)

    digest = advisor.calculations_digest([evidence])
    lambda_label = next(label for label, value in digest[0]["results"].items()
                        if value == "1/10")
    assert {"fraction", "pressure", "removed"} <= set(lambda_label.casefold().split())
    assert set(digest[0]["results"].values()) == {"1/2", "3/4", "1/10"}


