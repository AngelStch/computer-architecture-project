#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
archsim.py - симулационен модел към курсов проект по Компютърни архитектури.

Модули:
  1. Закони на Амдал и Густафсон
  2. Трасово-управляем симулатор на кеш памет (LRU, произволна асоциативност)
  3. Симулатор на предсказатели на преходи (статичен, 1-битов, 2-битов, gshare, турнирен)
  4. Модел на ефективния CPI и времето за изпълнение

Автор: Ангел Димитров Стойчев, фак. No 121224030
Стартиране: python3 archsim.py
"""
import json
import random
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(121224030)          # възпроизводимост на резултатите
OUT = "."

# --------------------------------------------------------------------------
# 1. Закони на Амдал и Густафсон
# --------------------------------------------------------------------------
def amdahl(f, n):
    """Ускорение при фиксиран обем на задачата; f - паралелна част."""
    return 1.0 / ((1.0 - f) + f / n)


def gustafson(f, n):
    """Мащабирано ускорение при растящ с n обем на задачата."""
    return (1.0 - f) + f * n


# --------------------------------------------------------------------------
# 2. Симулатор на кеш памет
# --------------------------------------------------------------------------
class Cache:
    """Кеш с множествена асоциативност и заместване по LRU."""

    def __init__(self, size, block, assoc):
        self.block = block
        self.assoc = assoc
        self.nsets = size // (block * assoc)
        self.sets = [OrderedDict() for _ in range(self.nsets)]
        self.hits = 0
        self.misses = 0

    def access(self, addr):
        blk = addr // self.block
        s = self.sets[blk % self.nsets]
        tag = blk // self.nsets
        if tag in s:
            s.move_to_end(tag)          # най-скоро използван
            self.hits += 1
            return True
        self.misses += 1
        if len(s) >= self.assoc:
            s.popitem(last=False)       # изхвърля най-отдавна използвания
        s[tag] = True
        return False

    @property
    def miss_rate(self):
        return self.misses / (self.hits + self.misses)


def run_cache(trace, size, block=64, assoc=4):
    c = Cache(size, block, assoc)
    for a in trace:
        c.access(a)
    return c.miss_rate


# --- генератори на адресни трасета ---------------------------------------
ELEM = 8                                   # double = 8 байта
MAT_N = 72                                 # размерност на матриците
A_BASE = 0x100000                          # матриците са разположени една след друга
B_BASE = A_BASE + MAT_N * MAT_N * ELEM + 192
C_BASE = B_BASE + MAT_N * MAT_N * ELEM + 320


def trace_matmul_ijk(n):
    """C[i][j] += A[i][k] * B[k][j]; B се обхожда по колони."""
    t = []
    for i in range(n):
        for j in range(n):
            t.append(C_BASE + (i * n + j) * ELEM)
            for k in range(n):
                t.append(A_BASE + (i * n + k) * ELEM)
                t.append(B_BASE + (k * n + j) * ELEM)
    return t


def trace_matmul_ikj(n):
    """Същото умножение с разменени цикли; B и C се обхождат по редове."""
    t = []
    for i in range(n):
        for k in range(n):
            t.append(A_BASE + (i * n + k) * ELEM)
            for j in range(n):
                t.append(B_BASE + (k * n + j) * ELEM)
                t.append(C_BASE + (i * n + j) * ELEM)
    return t


def trace_rowsum(n):
    return [A_BASE + (i * n + j) * ELEM for i in range(n) for j in range(n)]


def trace_colsum(n):
    return [A_BASE + (i * n + j) * ELEM for j in range(n) for i in range(n)]


def trace_vecsum(n, align):
    """c[i] = a[i] + b[i]; трите вектора са подравнени на кратно на align."""
    t = []
    for i in range(n):
        t.append(A_BASE + i * ELEM)
        t.append(A_BASE + align + i * ELEM)
        t.append(A_BASE + 2 * align + i * ELEM)
    return t


def trace_random(count, span):
    return [A_BASE + random.randrange(span // ELEM) * ELEM for _ in range(count)]


# --------------------------------------------------------------------------
# 3. Предсказатели на преходи
# --------------------------------------------------------------------------
class AlwaysTaken:
    name = "Статичен (винаги изпълнен)"

    def predict(self, pc):
        return True

    def update(self, pc, taken):
        pass


class OneBit:
    name = "1-битов"

    def __init__(self, bits=12):
        self.mask = (1 << bits) - 1
        self.t = [1] * (1 << bits)

    def predict(self, pc):
        return self.t[pc & self.mask] == 1

    def update(self, pc, taken):
        self.t[pc & self.mask] = 1 if taken else 0


class TwoBit:
    """Бимодален предсказател с 2-битови наситени броячи."""
    name = "2-битов (бимодален)"

    def __init__(self, bits=12):
        self.mask = (1 << bits) - 1
        self.t = [2] * (1 << bits)

    def _idx(self, pc):
        return pc & self.mask

    def predict(self, pc):
        return self.t[self._idx(pc)] >= 2

    def update(self, pc, taken):
        i = self._idx(pc)
        if taken:
            self.t[i] = min(3, self.t[i] + 1)
        else:
            self.t[i] = max(0, self.t[i] - 1)


class GShare(TwoBit):
    """Глобална история, комбинирана с адреса чрез XOR."""
    name = "gshare"

    def __init__(self, bits=12, hist=12):
        super().__init__(bits)
        self.hmask = (1 << hist) - 1
        self.ghr = 0

    def _idx(self, pc):
        return (pc ^ self.ghr) & self.mask

    def update(self, pc, taken):
        super().update(pc, taken)
        self.ghr = ((self.ghr << 1) | int(taken)) & self.hmask


class Tournament:
    """Избира между бимодален и gshare чрез таблица от 2-битови селектори."""
    name = "Турнирен"

    def __init__(self, bits=12, hist=12):
        self.p1 = TwoBit(bits)
        self.p2 = GShare(bits, hist)
        self.mask = (1 << bits) - 1
        self.sel = [2] * (1 << bits)

    def predict(self, pc):
        self._a = self.p1.predict(pc)
        self._b = self.p2.predict(pc)
        return self._b if self.sel[pc & self.mask] >= 2 else self._a

    def update(self, pc, taken):
        i = pc & self.mask
        if self._a != self._b:
            if self._b == taken:
                self.sel[i] = min(3, self.sel[i] + 1)
            else:
                self.sel[i] = max(0, self.sel[i] - 1)
        self.p1.update(pc, taken)
        self.p2.update(pc, taken)


def run_predictor(pred, trace):
    wrong = 0
    for pc, taken in trace:
        if pred.predict(pc) != taken:
            wrong += 1
        pred.update(pc, taken)
    return 1.0 - wrong / len(trace)


# --- трасета на преходи, получени чрез реално изпълнение на алгоритми -----
def btrace_bubble(n):
    a = [random.random() for _ in range(n)]
    t = []
    for i in range(n - 1):
        for j in range(n - 1 - i):
            sw = a[j] > a[j + 1]
            t.append((0x40, sw))                    # преход по данни
            if sw:
                a[j], a[j + 1] = a[j + 1], a[j]
            t.append((0x48, j + 1 < n - 1 - i))     # преход на вътрешния цикъл
        t.append((0x50, i + 1 < n - 1))             # преход на външния цикъл
    return t


def btrace_bsearch(n, q):
    a = sorted(random.randrange(10 * n) for _ in range(n))
    t = []
    for _ in range(q):
        key = random.randrange(10 * n)
        lo, hi = 0, n - 1
        while True:
            go = lo <= hi
            t.append((0x80, go))
            if not go:
                break
            mid = (lo + hi) // 2
            eq = a[mid] == key
            t.append((0x88, eq))
            if eq:
                break
            less = a[mid] < key
            t.append((0x90, less))                  # практически случаен
            if less:
                lo = mid + 1
            else:
                hi = mid - 1
    return t


def btrace_loops(outer, inner):
    """Вложени цикли с къс вътрешен цикъл (типично за обработка на матрици)."""
    t = []
    for i in range(outer):
        for j in range(inner):
            t.append((0xC0, j + 1 < inner))
        t.append((0xC8, i + 1 < outer))
    return t


def btrace_pattern(n):
    """Периодични и корелирани преходи."""
    t = []
    for i in range(n):
        a = i % 2 == 0
        b = i % 3 == 0
        t.append((0x100, a))
        t.append((0x108, b))
        t.append((0x110, a and b))                  # зависи от предходните два
        t.append((0x118, i % 5 != 4))
    return t


# --------------------------------------------------------------------------
# 4. Модел на ефективния CPI
# --------------------------------------------------------------------------
def effective_cpi(base, f_branch, mispredict, pen_branch, f_mem, miss, pen_miss):
    return base + f_branch * mispredict * pen_branch + f_mem * miss * pen_miss


# --------------------------------------------------------------------------
# Експерименти
# --------------------------------------------------------------------------
def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 150})


def exp_amdahl(res):
    ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    fs = [0.50, 0.75, 0.90, 0.95, 0.99]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for f in fs:
        ax[0].plot(ns, [amdahl(f, n) for n in ns], marker="o", ms=3, label=f"f = {f:.2f}")
        ax[1].plot(ns, [gustafson(f, n) for n in ns], marker="o", ms=3, label=f"f = {f:.2f}")
    for a, title in zip(ax, ["Закон на Амдал", "Закон на Густафсон"]):
        a.set_xscale("log", base=2)
        a.set_xlabel("Брой процесори N")
        a.set_ylabel("Ускорение S")
        a.set_title(title)
        a.legend(fontsize=9)
    ax[1].set_yscale("log", base=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_amdahl.png")
    res["amdahl"] = {f"{f:.2f}": {str(n): round(amdahl(f, n), 2) for n in (4, 16, 64, 1024)} for f in fs}
    res["gustafson"] = {f"{f:.2f}": {str(n): round(gustafson(f, n), 2) for n in (4, 16, 64, 1024)} for f in fs}


def exp_cache(res):
    n = MAT_N
    traces = {
        "Умножение ijk": trace_matmul_ijk(n),
        "Умножение ikj": trace_matmul_ikj(n),
    }
    res["trace_len"] = {k: len(v) for k, v in traces.items()}

    # 2.1 Размер на кеша
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    r = {}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, tr in traces.items():
        r[name] = [round(100 * run_cache(tr, s * 1024, 64, 4), 2) for s in sizes]
        ax.plot(sizes, r[name], marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Размер на кеша, KB (4-пътен, блок 64 B)")
    ax.set_ylabel("Дял на пропуските, %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cache_size.png")
    res["cache_size"] = {"sizes_kb": sizes, **r}

    # 2.2 Асоциативност
    assocs = [1, 2, 4, 8]
    r = {
        "Сума на вектори (подравнени)": [round(100 * run_cache(trace_vecsum(20000, 1 << 16), 32 * 1024, 64, a), 2) for a in assocs],
        "Умножение ijk (кеш 8 KB)": [round(100 * run_cache(traces["Умножение ijk"], 8 * 1024, 64, a), 2) for a in assocs],
    }
    res["cache_assoc"] = {"assoc": assocs, **r}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    w = 0.38
    xs = range(len(assocs))
    for k, (name, vals) in enumerate(r.items()):
        ax.bar([x + (k - 0.5) * w for x in xs], vals, w, label=name)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{a}-пътен" if a > 1 else "Директен" for a in assocs])
    ax.set_ylabel("Дял на пропуските, %")
    ax.set_xlabel("Асоциативност (блок 64 B)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cache_assoc.png")

    # 2.3 Размер на блока
    blocks = [16, 32, 64, 128, 256]
    tr2 = {
        "Последователно обхождане": trace_rowsum(256),
        "Умножение ijk": traces["Умножение ijk"],
        "Случаен достъп (1 MB)": trace_random(200000, 1 << 20),
    }
    r = {}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, tr in tr2.items():
        r[name] = [round(100 * run_cache(tr, 16 * 1024, b, 4), 2) for b in blocks]
        ax.plot(blocks, r[name], marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("Размер на блока, B (кеш 16 KB, 4-пътен)")
    ax.set_ylabel("Дял на пропуските, %")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cache_block.png")
    res["cache_block"] = {"blocks": blocks, **r}

    # 2.4 Ред на обхождане на двумерен масив 1000 x 1000
    row, col = trace_rowsum(1000), trace_colsum(1000)
    res["traversal"] = {
        "row": round(100 * run_cache(row, 32 * 1024, 64, 8), 2),
        "col": round(100 * run_cache(col, 32 * 1024, 64, 8), 2),
    }


def exp_branch(res):
    traces = {
        "Сортиране (метод на мехурчето)": btrace_bubble(300),
        "Двоично търсене": btrace_bsearch(1024, 4000),
        "Вложени цикли (8 итерации)": btrace_loops(10000, 8),
        "Периодични и корелирани": btrace_pattern(20000),
    }
    res["btrace_len"] = {k: len(v) for k, v in traces.items()}
    res["btrace_taken"] = {k: round(100 * sum(t for _, t in v) / len(v), 2) for k, v in traces.items()}
    makers = [AlwaysTaken, OneBit, TwoBit, GShare, Tournament]
    table = {}
    for name, tr in traces.items():
        table[name] = [round(100 * run_predictor(m(), tr), 2) for m in makers]
    res["branch"] = {"predictors": [m.name for m in makers], **table}

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    w = 0.16
    names = list(traces)
    for k, m in enumerate(makers):
        ax.bar([x + (k - 2) * w for x in range(len(names))],
               [table[n][k] for n in names], w, label=m.name)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["Сортиране", "Двоично\nтърсене", "Вложени\nцикли", "Периодични/\nкорелирани"])
    ax.set_ylabel("Точност на предсказване, %")
    ax.set_ylim(40, 101)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_branch.png")

    # дължина на историята при gshare
    hs = [0, 1, 2, 4, 6, 8, 10, 12]
    r = {}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in ("Вложени цикли (8 итерации)", "Периодични и корелирани", "Сортиране (метод на мехурчето)"):
        r[name] = [round(100 * run_predictor(GShare(12, h), traces[name]), 2) for h in hs]
        ax.plot(hs, r[name], marker="o", label=name)
    ax.set_xlabel("Дължина на глобалната история, бита")
    ax.set_ylabel("Точност на предсказване, %")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_gshare_hist.png")
    res["gshare_hist"] = {"hist": hs, **r}


def exp_cpi(res):
    """Чувствителност на CPI към точността на предсказване и пропуските в кеша."""
    base, f_br, pen_br, f_mem, pen_miss = 0.25, 0.20, 16, 0.35, 12
    acc = res["branch"]["Сортиране (метод на мехурчето)"]
    preds = res["branch"]["predictors"]
    miss_ijk = res["cache_size"]["Умножение ijk"][2] / 100      # кеш 4 KB
    miss_ikj = res["cache_size"]["Умножение ikj"][2] / 100
    rows = []
    for p, a in zip(preds, acc):
        for label, miss in (("ijk", miss_ijk), ("ikj", miss_ikj)):
            cpi = effective_cpi(base, f_br, 1 - a / 100, pen_br, f_mem, miss, pen_miss)
            rows.append({"pred": p, "order": label, "cpi": round(cpi, 3),
                         "ipc": round(1 / cpi, 2),
                         "time_ms": round(1e9 * cpi / 4e9 * 1e3, 1)})
    res["cpi"] = {"params": {"base": base, "f_br": f_br, "pen_br": pen_br,
                             "f_mem": f_mem, "pen_miss": pen_miss,
                             "ic": 1e9, "freq_ghz": 4.0}, "rows": rows}

    fig, ax = plt.subplots(figsize=(7.2, 4))
    w = 0.38
    for k, o in enumerate(("ijk", "ikj")):
        vals = [r["cpi"] for r in rows if r["order"] == o]
        ax.bar([x + (k - 0.5) * w for x in range(len(preds))], vals, w,
               label=f"Ред на циклите {o}")
    ax.set_xticks(range(len(preds)))
    ax.set_xticklabels(["Статичен", "1-битов", "2-битов", "gshare", "Турнирен"])
    ax.set_ylabel("Ефективен CPI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cpi.png")


def main():
    style()
    res = {}
    exp_amdahl(res)
    exp_cache(res)
    exp_branch(res)
    exp_cpi(res)
    with open(f"{OUT}/results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
