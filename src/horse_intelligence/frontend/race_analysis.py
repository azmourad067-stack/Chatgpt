
"""Analyse approfondie d'une course : probabilités, décision et comparaison au marché."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from horse_intelligence.frontend.components.cards import (
    ELECTRIC,
    EMERALD,
    metric_card,
    notice,
    status_badge,
)
from horse_intelligence.frontend.state import format_dt, get_context
from horse_intelligence.prediction.race_predictor import (
    LoadedFamily,
    RacePrediction,
    build_frame,
    load_models,
    models_use_source_aggregates,
    predict_race,
)

MODEL_LABELS = {
    "lgbm_market": "LightGBM + marché",
    "lgbm_form": "LightGBM (forme seule)",
    "market_calibrated": "Marché calibré",
}


@st.cache_resource(show_spinner=False)
def _models(_version_key: str) -> dict[str, LoadedFamily]:
    return load_models(get_context().repo.engine)


@st.cache_resource(show_spinner=False, max_entries=3)
def _frame(day: date, _data_key: str, use_agg: bool) -> pd.DataFrame:
    return build_frame(get_context().repo.engine, day, use_agg)


def _model_key() -> str:
    from horse_intelligence.models.registry import latest_model

    engine = get_context().repo.engine
    metas = [
        latest_model(engine, f)
        for f in ("lgbm_market", "lgbm_form", "market_calibrated")
    ]
    return "|".join(str(m["version"]) if m else "-" for m in metas)


def _predict(race_id: int, day: date) -> RacePrediction:
    engine = get_context().repo.engine
    mkey = _model_key()
    models = _models(mkey)

    if not models:
        return predict_race(engine, race_id)

    data_key = f"{mkey}|{get_context().repo.last_update()}"
    frame = _frame(
        day,
        data_key,
        models_use_source_aggregates(models),
    )

    return predict_race(
        engine,
        race_id,
        frame=frame,
        models=models,
    )


def _pct(p: float | None) -> str:
    return "—" if p is None or pd.isna(p) else f"{100 * p:.1f} %"


def _render_decision(pred: RacePrediction, table: pd.DataFrame) -> None:
    """Affiche la décision SIMPLE_GAGNANT et ses justifications."""

    st.subheader("Décision SIMPLE_GAGNANT")

    decision = pred.decision

    if decision is None:
        notice("Aucune décision disponible pour cette course.")
        return

    if decision.decision == "BET":
        st.info(
            f"**BET — candidat détecté : cheval n° {decision.num_pmu}**\n\n"
            f"{decision.reason}"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Probabilité estimée", _pct(decision.p_win))

        with c2:
            metric_card("Borne basse", _pct(decision.p_low))

        with c3:
            metric_card("Avantage conservateur", _pct(decision.conservative_edge))

        with c4:
            value = decision.conservative_ev
            metric_card(
                "Espérance théorique",
                "—" if value is None else f"{value:+.3f} €",
                "pour 1 € théorique",
            )

        horse_row = table[
            table["num_pmu"] == decision.num_pmu
        ]

        if not horse_row.empty:
            horse_name = horse_row.iloc[0].get("cheval", "Nom indisponible")
            st.caption(
                f"Cheval retenu : n° {decision.num_pmu} — {horse_name}"
            )

    elif decision.decision == "NO_BET":
        st.warning(
            f"**NO BET**\n\n{decision.reason}"
        )

    else:
        st.info(
            f"**INSUFFICIENT DATA — abstention obligatoire**\n\n"
            f"{decision.reason}"
        )

    st.caption(
        "Décision expérimentale fondée sur les données et les seuils du modèle. "
        "BET ne signifie pas qu'un pari est rentable ou qu'un gain est garanti. "
        "Aucun pari réel ne doit être déclenché automatiquement."
    )


def _render_prediction(
    pred: RacePrediction,
    horses: dict[int, str],
) -> None:
    st.markdown(
        status_badge(pred.status),
        unsafe_allow_html=True,
    )

    if pred.table.empty:
        notice(pred.message)
        return

    meta = pred.models[pred.primary or ""]

    if meta["status"] == "active":
        st.success(pred.message)
    else:
        st.warning(pred.message)

    unc_text = {
        "exploitable": "exploitable",
        "incertaine": "incertaine",
        "hors distribution": "hors distribution",
        "non évaluée": "non évaluée",
    }

    if pred.uncertainty in ("incertaine", "hors distribution"):
        st.warning(
            f"Course {unc_text[pred.uncertainty]} : "
            + " ; ".join(pred.uncertainty_reasons)
            + ". Aucun signal exploitable n'est produit."
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Modèle principal",
            MODEL_LABELS.get(pred.primary or "", "?"),
            f"version {meta['version']} · {meta['status']}",
        )

    with c2:
        metric_card(
            "Places payées",
            str(pred.n_places) if pred.n_places else "aucune",
            "simple placé, selon le nombre de partants",
        )

    with c3:
        metric_card(
            "Cotes de référence",
            "disponibles" if pred.has_market else "absentes",
            "marché utilisé comme référence",
        )

    with c4:
        metric_card(
            "Incertitude",
            unc_text.get(pred.uncertainty, pred.uncertainty),
            "ensemble bootstrap et profil de course",
        )

    t = pred.table.copy()
    t["cheval"] = t["horse_id"].map(horses)

    # Affichage de la décision avant le tableau des partants.
    _render_decision(pred, t)

    view = pd.DataFrame({
        "Rang": t["rank"].astype(int),
        "N°": t["num_pmu"],
        "Cheval": t["cheval"],
        "P(victoire)": t["p_win"].map(_pct),
        "Intervalle": [
            f"{_pct(a)} – {_pct(b)}"
            for a, b in zip(t["p_low"], t["p_high"], strict=True)
        ],
        "P(place)": t["p_place"].map(_pct),
        "P(marché)": t["mkt_prob"].map(_pct),
        "Cote réf.": t["mkt_odds"].map(
            lambda x: "—" if pd.isna(x) else f"{x:.1f}"
        ),
        "Désaccord": t["disagreement"].map(_pct),
        "Musique": t["form_string"],
    })

    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Probabilités arrondies : leur précision réelle est moindre. "
        "Intervalle = 10e-90e centiles d'un ensemble de modèles appris "
        "sur des rééchantillonnages de courses. Désaccord = écart maximal "
        "entre les estimations des modèles et du marché."
    )

    st.subheader("Forces, faiblesses et signaux faibles")

    for r in t.to_dict("records"):
        with st.expander(
            f"N° {r['num_pmu']} — {r['cheval'] or '?'} · {_pct(r['p_win'])}"
        ):
            st.markdown(f"**Forces** : {r['strengths'] or '—'}")
            st.markdown(f"**Faiblesses** : {r['weaknesses'] or '—'}")
            st.markdown(
                f"**Signaux faibles validés** : "
                f"{r['weak_signals'] or 'aucun'}"
            )

    st.caption(
        "Forces et faiblesses : variables qui font monter ou baisser "
        "l'estimation par rapport aux autres partants. Elles décrivent "
        "le modèle, pas une cause. Signaux faibles : uniquement les "
        "patterns validés hors échantillon par le Laboratoire IA."
    )

    fig = go.Figure()

    fig.add_bar(
        name="Modèle",
        x=t["num_pmu"].astype(str),
        y=t["p_win"],
        marker_color=EMERALD,
    )

    if pred.has_market:
        fig.add_bar(
            name="Marché",
            x=t["num_pmu"].astype(str),
            y=t["mkt_prob"],
            marker_color=ELECTRIC,
        )

    fig.update_layout(
        barmode="group",
        height=300,
        template="plotly_dark",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_tickformat=".0%",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="N°",
    )

    st.plotly_chart(fig, width="stretch")

    if pred.has_market:
        st.subheader("Comparaison avec les cotes")

        flagged = t[t["value_signal"]]

        if meta["status"] != "active":
            notice(
                "Signal de valeur désactivé : le modèle n'est pas validé. "
                "Un écart avec le marché n'a pas de valeur démontrée."
            )

        elif flagged.empty:
            notice(
                "Aucun écart supérieur à l'incertitude des estimations."
            )

        else:
            st.dataframe(
                pd.DataFrame({
                    "N°": flagged["num_pmu"],
                    "Cheval": flagged["cheval"],
                    "Écart": flagged["edge"].map(_pct),
                    "Espérance théorique (1 €)": flagged[
                        "ev_theoretical"
                    ].map(lambda x: f"{x:+.2f} €"),
                }),
                hide_index=True,
                width="stretch",
            )

        st.caption(
            "Espérance calculée sur la cote de référence ; au pari mutuel, "
            "le rapport final diffère. Une espérance estimée positive "
            "n'est jamais une garantie."
        )


def render() -> None:
    st.title("Analyse approfondie")

    repo = get_context().repo
    dates = repo.available_dates()

    if not dates:
        notice("Aucune course en base.")
        return

    day = st.date_input(
        "Date",
        value=dates[0],
        min_value=dates[-1],
        max_value=dates[0],
        format="DD/MM/YYYY",
        key="analysis_date",
    )

    races = repo.races_on(
        day if isinstance(day, date) else dates[0]
    )

    if races.empty:
        notice("Aucune course collectée pour cette date.")
        return

    labels = {
        int(r["race_id"]): (
            f"R{r['reunion_num']}C{r['course_num']} — "
            f"{r['racecourse'] or ''} — {r['label'] or ''}"
        )
        for r in races.to_dict("records")
    }

    race_id = st.selectbox(
        "Course",
        list(labels),
        format_func=labels.__getitem__,
    )

    row = races[races["race_id"] == race_id].iloc[0]

    st.caption(
        f"{row['discipline'] or '?'} · "
        f"{row['distance_m'] or '?'} m · "
        f"départ {format_dt(row['scheduled_start_utc'])}"
    )

    entries = repo.entries_as_of(int(race_id))

    horses = (
        dict(zip(entries["horse_id"], entries["horse"], strict=True))
        if not entries.empty
        else {}
    )

    with st.spinner("Calcul des probabilités…"):
        pred = _predict(
            int(race_id),
            day if isinstance(day, date) else dates[0],
        )

    _render_prediction(pred, horses)
