from __future__ import annotations
from typing import Optional, List
from datetime import date, datetime

import pandas as pd
import streamlit as st
from sqlmodel import SQLModel, Field, create_engine, Session, select

# ---------- DB ----------
class Transaction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    date: date
    type: str = "Покупка"           # Покупка/Продажа/Купон/Дивиденды/Внесение/Вывод/Прочее
    ticker: Optional[str] = None
    asset: Optional[str] = None
    portfolio: Optional[str] = "Основной"
    currency: str = "RUB"
    qty: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    amount: Optional[float] = None  # если не задано — посчитаем
    note: Optional[str] = None

DB_URL = "sqlite:///transactions.db"
engine = create_engine(DB_URL, echo=False)
SQLModel.metadata.create_all(engine)

# ---------- Helpers ----------
def row_to_dict(o):
    # SQLModel(Pydantic v1/v2) совместимость
    if hasattr(o, "model_dump"):  # pydantic v2
        return o.model_dump()
    return o.dict()               # pydantic v1

def signed_amount(qty: float, price: float, fee: float, kind: str) -> float:
    base = qty * price
    k = (kind or "").lower()
    if "покуп" in k or "buy" in k:
        return -(base + abs(fee))
    if "прод" in k or "sell" in k:
        return base - abs(fee)
    # купоны/дивиденды/внесение/вывод/прочее — как есть минус комиссия
    return base - abs(fee)

def fetch_df(filters: dict | None = None) -> pd.DataFrame:
    with Session(engine) as s:
        q = select(Transaction)
        rows = s.exec(q).all()
    df = pd.DataFrame([row_to_dict(r) for r in rows]) if rows else pd.DataFrame(
        columns=["id","date","type","ticker","asset","portfolio","currency","qty","price","fee","amount","note"]
    )
    if not df.empty and filters:
        if "date" in df.columns and filters.get("d1") and filters.get("d2"):
            df = df[(pd.to_datetime(df["date"]).dt.date >= filters["d1"]) &
                    (pd.to_datetime(df["date"]).dt.date <= filters["d2"])]
        for col in ["type","currency","portfolio"]:
            if filters.get(col):
                df = df[df[col].isin(filters[col])]
        if txt := filters.get("search"):
            txt = txt.lower()
            mask = False
            for c in [c for c in ["ticker","asset","note"] if c in df]:
                mask = mask | df[c].astype(str).str.lower().str.contains(txt, na=False)
            df = df[mask] if isinstance(mask, pd.Series) else df
    return df.sort_values(by="date", ascending=False)

def insert_tx(tx: Transaction):
    with Session(engine) as s:
        s.add(tx)
        s.commit()

def update_tx(tx_id: int, data: dict):
    with Session(engine) as s:
        obj = s.get(Transaction, tx_id)
        if not obj: return
        for k,v in data.items():
            setattr(obj, k, v)
        s.add(obj)
        s.commit()

def delete_ids(ids: List[int]):
    if not ids: return
    with Session(engine) as s:
        for i in ids:
            obj = s.get(Transaction, i)
            if obj: s.delete(obj)
        s.commit()

# ---------- UI ----------
st.set_page_config(page_title="Книга транзакций", layout="wide")
st.title("Книга транзакций")

# --- Filters ---
df_all = fetch_df()
if df_all.empty:
    mind, maxd = date.today(), date.today()
else:
    mind, maxd = pd.to_datetime(df_all["date"]).min().date(), pd.to_datetime(df_all["date"]).max().date()

with st.sidebar:
    st.header("Фильтры")
    d1, d2 = st.date_input("Период", (mind, maxd))
    types_opts = sorted(df_all["type"].dropna().unique().tolist()) if not df_all.empty else []
    curr_opts  = sorted(df_all["currency"].dropna().unique().tolist()) if not df_all.empty else ["RUB","USD","EUR"]
    port_opts  = sorted(df_all["portfolio"].dropna().unique().tolist()) if not df_all.empty else ["Основной"]
    types = st.multiselect("Тип операции", types_opts, default=types_opts)
    currs = st.multiselect("Валюта", curr_opts, default=curr_opts)
    ports = st.multiselect("Портфель", port_opts, default=port_opts)
    search = st.text_input("Поиск (тикер/инструмент/заметка)")
    filters = {"d1": d1, "d2": d2, "type": types, "currency": currs, "portfolio": ports, "search": search}

df = fetch_df(filters)

# --- KPI ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Всего сделок", len(df))
k2.metric("Комиссии", f'{(df["fee"].sum() if "fee" in df else 0):,.2f}'.replace(",", " "))
k3.metric("Сумма покупок", f'{df.query("type.str.contains(\'покуп\', case=False)", engine="python")["amount"].sum() if not df.empty and "amount" in df else 0:,.2f}'.replace(",", " "))
k4.metric("Сумма продаж", f'{df.query("type.str.contains(\'прод\', case=False)", engine="python")["amount"].sum() if not df.empty and "amount" in df else 0:,.2f}'.replace(",", " "))

st.divider()

# --- Add new ---
with st.expander("Добавить транзакцию", expanded=True):
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        dt = c1.date_input("Дата", value=date.today())
        typ = c2.selectbox("Тип", ["Покупка","Продажа","Купон","Дивиденды","Внесение","Вывод","Прочее"])
        cur = c3.selectbox("Валюта", ["RUB","USD","EUR"])
        port = c4.text_input("Портфель", value="Основной")

        c5, c6, c7, c8 = st.columns(4)
        ticker = c5.text_input("Тикер/ISIN", value="")
        asset  = c6.text_input("Инструмент", value="")
        qty    = c7.number_input("Количество", value=0.0, step=1.0, format="%.4f")
        price  = c8.number_input("Цена", value=0.0, step=0.01, format="%.6f")

        c9, c10, c11 = st.columns(3)
        fee    = c9.number_input("Комиссия", value=0.0, step=0.01)
        amount = c10.number_input("Сумма (опц.)", value=0.0, step=0.01,
                                  help="Если оставить 0 — посчитаем автоматически по qty*price с учётом типа и комиссии.")
        note   = c11.text_input("Заметка", value="")

        ok = st.form_submit_button("Добавить")
        if ok:
            amt = None if amount == 0 else amount
            if amt is None:
                amt = signed_amount(qty, price, fee, typ)
            insert_tx(Transaction(date=dt, type=typ, ticker=ticker, asset=asset,
                                  portfolio=port, currency=cur, qty=qty, price=price,
                                  fee=fee, amount=amt, note=note))
            st.success("Добавлено")

# --- Table + actions ---
st.subheader("Транзакции")
if df.empty:
    st.info("Пока пусто. Добавьте первую запись выше.")
else:
    show = df.copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%d.%m.%Y")
    st.dataframe(show, use_container_width=True, hide_index=True)

    # редактирование
    st.markdown("### Редактирование/удаление")
    ids = df["id"].astype(int).tolist()
    col_a, col_b, col_c = st.columns([2,2,2])
    to_edit = col_a.selectbox("Выбрать запись для редактирования (ID)", ids)
    to_delete = col_b.multiselect("Удалить записи (ID)", [])
    if col_c.button("Удалить выбранные"):
        delete_ids(to_delete)
        st.success(f"Удалено: {to_delete}")

    # форма редактирования
    if to_edit:
        row = df[df["id"] == to_edit].iloc[0].to_dict()
        with st.form("edit_form"):
            e1, e2, e3, e4 = st.columns(4)
            dt = e1.date_input("Дата", value=pd.to_datetime(row["date"]).date())
            typ = e2.selectbox("Тип", ["Покупка","Продажа","Купон","Дивиденды","Внесение","Вывод","Прочее"], index=
                               ["Покупка","Продажа","Купон","Дивиденды","Внесение","Вывод","Прочее"].index(row["type"]) if row.get("type") in
                               ["Покупка","Продажа","Купон","Дивиденды","Внесение","Вывод","Прочее"] else 0)
            cur = e3.selectbox("Валюта", ["RUB","USD","EUR"], index= ["RUB","USD","EUR"].index(row.get("currency","RUB")) if row.get("currency") in ["RUB","USD","EUR"] else 0)
            port = e4.text_input("Портфель", value=row.get("portfolio") or "Основной")

            e5, e6, e7, e8 = st.columns(4)
            ticker = e5.text_input("Тикер/ISIN", value=row.get("ticker") or "")
            asset  = e6.text_input("Инструмент", value=row.get("asset") or "")
            qty    = e7.number_input("Количество", value=float(row.get("qty") or 0.0), step=1.0, format="%.4f")
            price  = e8.number_input("Цена", value=float(row.get("price") or 0.0), step=0.01, format="%.6f")

            e9, e10, e11 = st.columns(3)
            fee    = e9.number_input("Комиссия", value=float(row.get("fee") or 0.0), step=0.01)
            amount = e10.number_input("Сумма", value=float(row.get("amount") or 0.0), step=0.01)
            note   = e11.text_input("Заметка", value=row.get("note") or "")

            save = st.form_submit_button("Сохранить изменения")
            if save:
                update_tx(int(row["id"]), dict(date=dt, type=typ, currency=cur, portfolio=port,
                                               ticker=ticker, asset=asset, qty=qty, price=price,
                                               fee=fee, amount=amount, note=note))
                st.success("Сохранено")

# --- Export / Import ---
st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("Экспорт в CSV"):
        all_df = fetch_df({})
        st.download_button("Скачать CSV", all_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="transactions.csv", mime="text/csv")
with c2:
    upl = st.file_uploader("Импорт CSV (колонки: date,type,ticker,asset,portfolio,currency,qty,price,fee,amount,note)", type=["csv"])
    if upl is not None:
        imp = pd.read_csv(upl)
        # мягкое приведение
        for col in ["date","type","currency"]:
            if col not in imp: imp[col] = "" if col != "date" else date.today()
        imp["date"] = pd.to_datetime(imp["date"], dayfirst=True, errors="coerce").dt.date.fillna(date.today())
        imp = imp.fillna({"portfolio":"Основной","qty":0,"price":0,"fee":0,"amount":0,"note":""})
        for _, r in imp.iterrows():
            insert_tx(Transaction(**{k:r.get(k) for k in Transaction.model_fields.keys() if k!="id"}))
        st.success(f"Импортировано: {len(imp)}")
