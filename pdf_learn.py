# Install: pip install weasyprint

from weasyprint import HTML

# Your HTML content (paste it here)
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transaction Costs — Quant Finance Visual</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Clash+Display:wght@600;700&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@600;700&family=JetBrains+Mono:wght@400;500;700&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:       #F5F0E8;
  --bg2:      #EDE7D8;
  --panel:    #FFFFFF;
  --ink:      #1A1208;
  --ink2:     #3D3120;
  --muted:    #8A7D68;
  --border:   #D4C9B5;
  --border2:  #BFB09A;
  --red:      #C0392B;
  --orange:   #D4660A;
  --gold:     #B8860B;
  --teal:     #0E7A6E;
  --blue:     #1A5C8A;
  --purple:   #5C3D8A;
  --mono:     'JetBrains Mono', monospace;
  --head:     'Familjen Grotesk', sans-serif;
  --body:     'Fraunces', serif;
}

body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;min-height:100vh;overflow-x:hidden}

/* subtle dot grid */
body::before{
  content:'';position:fixed;inset:0;z-index:0;
  background-image:radial-gradient(var(--border) 1px,transparent 1px);
  background-size:28px 28px;pointer-events:none;opacity:0.5;
}

/* ── HEADER ── */
header{
  position:relative;z-index:1;
  background:var(--ink);
  padding:56px 72px 48px;overflow:hidden;
}
header::after{
  content:'TC';position:absolute;right:-20px;bottom:-40px;
  font-family:var(--head);font-size:280px;font-weight:700;
  color:rgba(255,255,255,0.03);line-height:1;pointer-events:none;
}
.h-eyebrow{
  font-family:var(--mono);font-size:10px;letter-spacing:4px;
  color:var(--gold);text-transform:uppercase;margin-bottom:18px;
  display:flex;align-items:center;gap:12px;
}
.h-eyebrow::before{content:'';width:32px;height:1px;background:var(--gold);}
h1{
  font-family:var(--head);font-size:44px;font-weight:700;
  color:#fff;line-height:1.05;letter-spacing:-1.5px;max-width:700px;
}
h1 em{color:#E8C56A;font-style:normal;}
.h-sub{margin-top:12px;font-size:16px;color:rgba(255,255,255,0.45);font-style:italic;}
.h-pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:26px;}
.pill{
  font-family:var(--mono);font-size:10px;letter-spacing:1px;
  padding:5px 13px;border-radius:2px;text-transform:uppercase;
  border:1px solid rgba(255,255,255,0.15);color:rgba(255,255,255,0.5);
}
.pill.hi{border-color:var(--gold);color:var(--gold);background:rgba(184,134,11,0.1);}

/* ── MAIN ── */
main{position:relative;z-index:1;padding:52px 72px;max-width:1380px;margin:0 auto;}

.sec-eyebrow{
  font-family:var(--mono);font-size:10px;letter-spacing:3px;
  text-transform:uppercase;color:var(--teal);margin-bottom:16px;
  display:flex;align-items:center;gap:10px;
  padding-bottom:10px;border-bottom:1px solid var(--border);
}
.sec-eyebrow span{color:var(--muted);}
.sec-h{
  font-family:var(--head);font-size:22px;font-weight:700;
  color:var(--ink);margin-bottom:26px;letter-spacing:-0.5px;
}
.block{margin-bottom:60px;}

/* ── COST ANATOMY ── */
.anatomy-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border-radius:12px;overflow:hidden;margin-bottom:28px;}
.ana-cell{background:var(--panel);padding:28px;transition:background 0.2s;}
.ana-cell:hover{background:#FAFAF7;}
.ana-num{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:10px;letter-spacing:2px;}
.ana-title{font-family:var(--head);font-size:15px;font-weight:700;color:var(--ink);margin-bottom:8px;}
.ana-body{font-size:13px;color:var(--ink2);line-height:1.75;}
.ana-formula{
  margin-top:12px;font-family:var(--mono);font-size:11px;
  padding:9px 13px;background:var(--bg2);
  border-left:3px solid var(--teal);color:var(--teal);
  border-radius:0 4px 4px 0;line-height:1.7;
}

/* ── WATERFALL ── */
.waterfall{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:36px;margin-bottom:28px;}
.wf-title{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--ink);margin-bottom:6px;}
.wf-sub{font-size:12px;color:var(--muted);margin-bottom:24px;font-style:italic;}
.wf-rows{display:flex;flex-direction:column;gap:10px;}
.wf-row{display:flex;align-items:center;gap:16px;}
.wf-label{font-family:var(--mono);font-size:11px;color:var(--ink2);width:180px;flex-shrink:0;text-align:right;}
.wf-bar-wrap{flex:1;position:relative;height:32px;background:var(--bg);border-radius:4px;overflow:hidden;}
.wf-bar{height:100%;border-radius:4px;display:flex;align-items:center;padding:0 10px;transition:width 0.8s ease;}
.wf-bar span{font-family:var(--mono);font-size:10px;font-weight:700;color:#fff;white-space:nowrap;}
.wf-val{font-family:var(--mono);font-size:11px;color:var(--muted);width:60px;flex-shrink:0;}

/* ── SIX COMPONENTS ── */
.comp-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:60px;}
.comp-card{
  background:var(--panel);border:1px solid var(--border);
  border-radius:10px;padding:24px;position:relative;overflow:hidden;
  transition:box-shadow 0.2s,transform 0.2s;
}
.comp-card:hover{box-shadow:0 8px 32px rgba(26,18,8,0.1);transform:translateY(-2px);}
.comp-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.cc-red::before    {background:var(--red);}
.cc-orange::before {background:var(--orange);}
.cc-gold::before   {background:var(--gold);}
.cc-teal::before   {background:var(--teal);}
.cc-blue::before   {background:var(--blue);}
.cc-purple::before {background:var(--purple);}
.comp-num{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:2px;margin-bottom:10px;}
.comp-name{font-family:var(--head);font-size:15px;font-weight:700;color:var(--ink);margin-bottom:6px;}
.comp-tag{
  font-family:var(--mono);font-size:10px;padding:2px 8px;border-radius:2px;
  display:inline-block;margin-bottom:14px;letter-spacing:0.5px;
}
.ct-explicit {background:rgba(12,122,110,0.1); color:var(--teal);}
.ct-implicit {background:rgba(212,102,10,0.1);  color:var(--orange);}
.ct-hidden   {background:rgba(92,61,138,0.1);   color:var(--purple);}
.comp-desc{font-size:13px;color:var(--ink2);line-height:1.7;margin-bottom:14px;}
.comp-formula{
  font-family:var(--mono);font-size:11px;padding:10px 13px;
  background:var(--bg2);border-left:2px solid var(--border2);
  color:var(--ink2);border-radius:0 4px 4px 0;line-height:1.6;
}
.comp-typical{
  margin-top:12px;display:flex;justify-content:space-between;
  align-items:center;padding-top:12px;border-top:1px solid var(--border);
}
.ct-range{font-family:var(--mono);font-size:11px;}
.ct-label{font-size:11px;color:var(--muted);}

/* ── TURNOVER DRAG ── */
.drag-section{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:60px;}
.drag-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:30px;}
.drag-title{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--ink);margin-bottom:20px;}

/* turnover matrix */
.matrix{width:100%;border-collapse:collapse;}
.matrix th{
  font-family:var(--mono);font-size:10px;letter-spacing:1px;
  text-transform:uppercase;padding:10px 14px;text-align:center;
  background:var(--ink);color:#fff;
}
.matrix th:first-child{text-align:left;border-radius:8px 0 0 0;}
.matrix th:last-child{border-radius:0 8px 0 0;}
.matrix td{
  padding:11px 14px;font-family:var(--mono);font-size:12px;
  text-align:center;border-bottom:1px solid var(--border);
  border-right:1px solid var(--border);
}
.matrix td:first-child{text-align:left;font-size:11px;color:var(--muted);font-weight:500;}
.matrix tr:last-child td{border-bottom:none;}
.matrix tr:last-child td:first-child{border-radius:0 0 0 8px;}
.td-lo{background:rgba(12,122,110,0.08);color:var(--teal);}
.td-md{background:rgba(184,134,11,0.10);color:var(--gold);}
.td-hi{background:rgba(192,57,43,0.10);color:var(--red);}
.td-ex{background:rgba(192,57,43,0.18);color:var(--red);font-weight:700;}

/* breakeven chart */
.be-chart{display:flex;flex-direction:column;gap:12px;margin-top:8px;}
.be-row{display:flex;align-items:center;gap:12px;}
.be-label{font-family:var(--mono);font-size:10px;color:var(--muted);width:120px;flex-shrink:0;}
.be-track{flex:1;height:24px;background:var(--bg);border-radius:3px;overflow:hidden;position:relative;}
.be-fill{height:100%;border-radius:3px;display:flex;align-items:center;justify-content:flex-end;padding-right:8px;}
.be-fill span{font-family:var(--mono);font-size:10px;font-weight:700;color:#fff;}
.be-note{font-family:var(--mono);font-size:10px;color:var(--muted);width:70px;flex-shrink:0;text-align:right;}

/* ── ASSET CLASS TABLE ── */
.asset-wrap{background:var(--panel);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:60px;}
.asset-table{width:100%;border-collapse:collapse;}
.asset-table th{
  background:var(--ink);color:#fff;font-family:var(--mono);font-size:10px;
  letter-spacing:2px;text-transform:uppercase;padding:14px 20px;text-align:left;
}
.asset-table td{
  padding:13px 20px;font-size:13px;border-bottom:1px solid var(--border);
  vertical-align:middle;
}
.asset-table tr:last-child td{border-bottom:none;}
.asset-table tr:hover td{background:#FAFAF5;}
.asset-name{font-family:var(--head);font-size:14px;font-weight:700;color:var(--ink);}
.dot-gauge{display:flex;gap:3px;align-items:center;}
.dg{width:9px;height:9px;border-radius:50%;}
.dg.fill{opacity:1;}
.dg.empty{background:var(--border)!important;opacity:1;}
.badge-sm{
  font-family:var(--mono);font-size:10px;padding:3px 8px;
  border-radius:2px;display:inline-block;
}
.b-lo {background:rgba(12,122,110,0.1); color:var(--teal);}
.b-md {background:rgba(184,134,11,0.12);color:var(--gold);}
.b-hi {background:rgba(192,57,43,0.1);  color:var(--red);}
.b-var{background:rgba(92,61,138,0.1);  color:var(--purple);}

/* ── IMPACT vs COST ── */
.impact-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:60px;}
.impact-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:26px;}
.ic-title{font-family:var(--head);font-size:14px;font-weight:700;color:var(--ink);margin-bottom:4px;}
.ic-sub{font-size:12px;color:var(--muted);font-style:italic;margin-bottom:16px;}
.ic-formula{
  font-family:var(--mono);font-size:11px;padding:10px 13px;
  background:var(--bg2);border-left:3px solid var(--teal);
  color:var(--teal);border-radius:0 4px 4px 0;margin-bottom:14px;line-height:1.65;
}
.ic-points{display:flex;flex-direction:column;gap:8px;}
.ic-point{display:flex;gap:8px;align-items:flex-start;font-size:12px;color:var(--ink2);line-height:1.55;}
.ic-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:4px;}

/* ── REGIME TABLE ── */
.regime-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border-radius:12px;overflow:hidden;margin-bottom:60px;}
.reg-cell{background:var(--panel);padding:22px;}
.reg-name{font-family:var(--head);font-size:14px;font-weight:700;color:var(--ink);margin-bottom:8px;}
.reg-mult{font-family:var(--mono);font-size:26px;font-weight:700;margin-bottom:8px;}
.reg-desc{font-size:12px;color:var(--muted);line-height:1.65;}
.rm-normal{color:var(--teal);}
.rm-stress{color:var(--gold);}
.rm-crisis{color:var(--orange);}
.rm-crash {color:var(--red);}

/* ── RED FLAGS ── */
.flags-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:60px;}
.flag{
  background:var(--panel);border:1px solid var(--border);
  border-radius:8px;padding:20px;border-left:3px solid var(--red);
}
.flag-name{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--ink);margin-bottom:6px;}
.flag-body{font-size:12px;color:var(--muted);line-height:1.6;}
.flag-fix{font-family:var(--mono);font-size:10px;color:var(--teal);margin-top:8px;}

/* ── CHECKLIST ── */
.cl-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:48px;}
.cl-item{
  display:flex;gap:10px;align-items:flex-start;
  background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:14px;
}
.cl-box{
  width:17px;height:17px;border-radius:3px;flex-shrink:0;
  border:1.5px solid var(--teal);color:var(--teal);
  display:flex;align-items:center;justify-content:center;
  font-size:10px;margin-top:1px;
}
.cl-text{font-size:12px;color:var(--ink2);line-height:1.55;}

/* ── FORMULA STRIP ── */
.formula-strip{
  background:var(--ink);border-radius:12px;
  padding:28px 36px;margin-bottom:60px;
  display:grid;grid-template-columns:repeat(3,1fr);gap:24px;
}
.fs-item{}
.fs-label{font-family:var(--mono);font-size:10px;letter-spacing:2px;color:rgba(255,255,255,0.35);text-transform:uppercase;margin-bottom:8px;}
.fs-formula{font-family:var(--mono);font-size:12px;color:#E8C56A;line-height:1.7;}
.fs-note{font-size:12px;color:rgba(255,255,255,0.4);margin-top:6px;font-style:italic;}

footer{
  background:var(--ink);color:rgba(255,255,255,0.25);
  text-align:center;padding:22px;
  font-family:var(--mono);font-size:10px;letter-spacing:3px;text-transform:uppercase;
}

@keyframes fadeUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
.block,.comp-grid,.drag-section,.impact-grid{animation:fadeUp 0.5s ease both;}
</style>
</head>
<body>

<header>
  <div class="h-eyebrow">Quant Finance · Theory Reference</div>
  <h1>Transaction <em>Costs</em><br>Visual Overview</h1>
  <p class="h-sub">Every cost component, its formula, how it compounds — at a glance</p>
  <div class="h-pills">
    <span class="pill hi">6 Cost Components</span>
    <span class="pill">Turnover Drag Matrix</span>
    <span class="pill">Asset Class Benchmarks</span>
    <span class="pill">Regime Multipliers</span>
    <span class="pill">Break-even Alpha</span>
    <span class="pill">Red Flags</span>
  </div>
</header>

<main>

<!-- ═══ WHAT ARE TRANSACTION COSTS ═══ -->
<div class="block">
  <div class="sec-eyebrow">§1 <span>/ Definition & Structure</span></div>
  <div class="sec-h">What Are Transaction Costs?</div>
  <div class="anatomy-grid">
    <div class="ana-cell">
      <div class="ana-num">01 · DEFINITION</div>
      <div class="ana-title">Total Execution Cost</div>
      <div class="ana-body">Every expense incurred when moving from one portfolio position to another. They are always a drag — they reduce gross alpha to net alpha. Understating them is the primary reason strategies fail in live trading after looking great in backtests.</div>
      <div class="ana-formula">Net Return = Gross Alpha − TC<br>TC = Spread + Impact + Commission<br>    + Taxes + Borrow + Opportunity</div>
    </div>
    <div class="ana-cell">
      <div class="ana-num">02 · THE SILENT COMPOUNDING</div>
      <div class="ana-title">Why They Destroy Strategies</div>
      <div class="ana-body">A 500% annual turnover strategy with 20 bps round-trip cost loses 100 bps per year to costs alone. At 1000% turnover, that is 200 bps. Paired with a 15% gross alpha, you are left with 13%. Mis-model by 2× and you are left with 11%. That gap, compounded, is the entire fund's edge.</div>
      <div class="ana-formula">Annual TC Drag = Turnover × RT_Cost<br>e.g. 500% × 20 bps = 1.0% per year</div>
    </div>
    <div class="ana-cell">
      <div class="ana-num">03 · EXPLICIT vs IMPLICIT</div>
      <div class="ana-title">Two Categories</div>
      <div class="ana-body"><strong>Explicit:</strong> Observable, invoiced directly — commission, exchange fees, taxes, SEC fee. Known before trading. Negotiable at scale.<br><br><strong>Implicit:</strong> Embedded in execution price — spread, market impact, opportunity cost, borrow. Not invoiced. Estimated through models. Often 3–5× larger than explicit costs.</div>
      <div class="ana-formula">Implicit costs dominate total TC<br>Typically 70–85% of total for<br>institutional equity strategies</div>
    </div>
  </div>

  <!-- WATERFALL -->
  <div class="waterfall">
    <div class="wf-title">From Gross Alpha to Net Return — The Cost Waterfall</div>
    <div class="wf-sub">Illustrative equity L/S strategy, 200% annual turnover, mid-cap universe</div>
    <div class="wf-rows">
      <div class="wf-row">
        <div class="wf-label">Gross Alpha</div>
        <div class="wf-bar-wrap"><div class="wf-bar" style="width:100%;background:#1A1208;"><span>15.0%</span></div></div>
        <div class="wf-val" style="color:var(--ink)">15.0%</div>
      </div>
      <div class="wf-row">
        <div class="wf-label">− Bid-Ask Spread</div>
        <div class="wf-bar-wrap"><div class="wf-bar" style="width:87%;background:var(--teal);"><span>−2.0%</span></div></div>
        <div class="wf-val" style="color:var(--teal)">−2.0%</div>
      </div>
      <div class="wf-row">
        <div class="wf-label">− Market Impact</div>
        <div class="wf-bar-wrap"><div class="wf-bar" style="width:74%;background:var(--gold);"><span>−2.0%</span></div></div>
        <div class="wf-val" style="color:var(--gold)">−2.0%</div>
      </div>
      <div class="wf-row">
        <div class="wf-label">− Commission + Fees</div>
        <div class="wf-bar-wrap"><div class="wf-bar" style="width:70%;background:var(--blue);"><span>−0.6%</span></div></div>
        <div class="wf-val" style="color:var(--blue)">−0.6%</div>
      </div>
      <div class="wf-row">
        <div class="wf-label">− Borrow Cost (short)</div>
        <div class="wf-bar-wrap"><div class="wf-bar" style="width:66%;background:var(--orange);"><span>−0.6%</span></div></div>
        <div class="wf-val" style="color:var(--orange)">−0.6%</div>
      </div>
      <div class="wf-row">
        <div class="wf-label">− Taxes + SEC fee</div>
        <div class="wf-bar-wrap"><div class="wf-bar" style="width:62%;background:var(--purple);"><span>−0.5%</span></div></div>
        <div class="wf-val" style="color:var(--purple)">−0.5%</div>
      </div>
      <div class="wf-row">
        <div class="wf-label" style="font-weight:700;color:var(--ink)">= Net Return</div>
        <div class="wf-bar-wrap"><div class="wf-bar" style="width:62%;background:var(--red);"><span>9.3%</span></div></div>
        <div class="wf-val" style="color:var(--red);font-weight:700">9.3%</div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ SIX COMPONENTS ═══ -->
<div class="block">
  <div class="sec-eyebrow">§2 <span>/ The Six Cost Components</span></div>
  <div class="sec-h">Every Cost Has a Formula and a Failure Mode</div>
</div>
<div class="comp-grid">

  <div class="comp-card cc-teal">
    <div class="comp-num">COMPONENT 01</div>
    <div class="comp-name">Bid-Ask Spread</div>
    <span class="comp-tag ct-implicit">IMPLICIT</span>
    <div class="comp-desc">Crossing from mid to ask (buy) or mid to bid (sell). Largest single cost for small, frequent trades. Time-varying — 2–5× wider at open/close and in stress periods.</div>
    <div class="comp-formula">Cost = ½ × (Ask − Bid)<br>     = ½ × Spread(t)<br><br>Annualised drag = Turnover × ½ Spread</div>
    <div class="comp-typical">
      <div class="ct-range" style="color:var(--teal)">2–30 bps</div>
      <div class="ct-label">typical per side, equity</div>
    </div>
  </div>

  <div class="comp-card cc-orange">
    <div class="comp-num">COMPONENT 02</div>
    <div class="comp-name">Market Impact</div>
    <span class="comp-tag ct-implicit">IMPLICIT</span>
    <div class="comp-desc">Your order moves the price against you as it is absorbed by the book. Sub-linear in order size. The dominant cost for institutional strategies with large positions.</div>
    <div class="comp-formula">Impact = σ × Y × √(Q / ADV)<br><br>σ = daily vol · Y ≈ 0.5–1.0<br>Q = size · ADV = avg daily vol</div>
    <div class="comp-typical">
      <div class="ct-range" style="color:var(--orange)">5–100+ bps</div>
      <div class="ct-label">depends on Q/ADV ratio</div>
    </div>
  </div>

  <div class="comp-card cc-blue">
    <div class="comp-num">COMPONENT 03</div>
    <div class="comp-name">Commission & Fees</div>
    <span class="comp-tag ct-explicit">EXPLICIT</span>
    <div class="comp-desc">Broker commission, exchange access fee, clearing fee. Fully explicit and invoiced. Negotiable at institutional scale. The smallest explicit cost — but still non-trivial at high turnover.</div>
    <div class="comp-formula">Commission = rate × notional<br>Rate: $0.001–$0.01/share<br>     or 0–5 bps of notional</div>
    <div class="comp-typical">
      <div class="ct-range" style="color:var(--blue)">0.5–5 bps</div>
      <div class="ct-label">per side, negotiated</div>
    </div>
  </div>

  <div class="comp-card cc-red">
    <div class="comp-num">COMPONENT 04</div>
    <div class="comp-name">Borrow Cost (Short)</div>
    <span class="comp-tag ct-explicit">EXPLICIT</span>
    <div class="comp-desc">Fee paid to borrow shares for short selling. Varies from 25 bps/yr (easy-to-borrow large caps) to 50%+ (hard-to-borrow squeeze targets). Often exceeds the short-side alpha signal entirely.</div>
    <div class="comp-formula">Annual Borrow Cost = borrow_rate × |short_notional|<br><br>GC rate: 0.25–0.50% pa<br>HTB rate: 5–50%+ pa</div>
    <div class="comp-typical">
      <div class="ct-range" style="color:var(--red)">0.25–50%+ pa</div>
      <div class="ct-label">per annum on short book</div>
    </div>
  </div>

  <div class="comp-card cc-purple">
    <div class="comp-num">COMPONENT 05</div>
    <div class="comp-name">Taxes & Regulatory Fees</div>
    <span class="comp-tag ct-explicit">EXPLICIT</span>
    <div class="comp-desc">SEC Section 31 fee on US equity sales. Financial transaction taxes (EU FTT, UK stamp duty 0.5%, France 0.3%). Short-term vs long-term capital gains differential. Jurisdiction-dependent.</div>
    <div class="comp-formula">SEC fee = 0.0000278 × sale_value<br>UK stamp duty = 0.005 × buy_value<br>France FTT = 0.003 × buy_value</div>
    <div class="comp-typical">
      <div class="ct-range" style="color:var(--purple)">0.003–50 bps</div>
      <div class="ct-label">varies by jurisdiction & asset</div>
    </div>
  </div>

  <div class="comp-card cc-gold">
    <div class="comp-num">COMPONENT 06</div>
    <div class="comp-name">Opportunity Cost</div>
    <span class="comp-tag ct-hidden">HIDDEN</span>
    <div class="comp-desc">Alpha lost on shares you intended to trade but couldn't execute (price moved before fill, order cancelled, insufficient liquidity). Part of Implementation Shortfall but rarely modelled explicitly.</div>
    <div class="comp-formula">Opp. Cost = (P_now − P_decision) × unexecuted_qty<br><br>IS = Explicit + Impact + Opp. Cost<br>IS = (P_exec − P_decision) × Q_filled</div>
    <div class="comp-typical">
      <div class="ct-range" style="color:var(--gold)">hard to quantify</div>
      <div class="ct-label">often 10–30% of total TC</div>
    </div>
  </div>

</div>

<!-- ═══ KEY FORMULAS ═══ -->
<div class="block">
  <div class="sec-eyebrow">§3 <span>/ Essential Formulas</span></div>
  <div class="sec-h">Numbers You Must Know Cold</div>
  <div class="formula-strip">
    <div class="fs-item">
      <div class="fs-label">Annual Cost Drag</div>
      <div class="fs-formula">TC_annual = Turnover × RT_Cost<br><br>e.g. 1000% × 20 bps = 2.00% pa</div>
      <div class="fs-note">RT = round-trip (entry + exit)</div>
    </div>
    <div class="fs-item">
      <div class="fs-label">Square Root Impact</div>
      <div class="fs-formula">Impact = σ × Y × √(Q / ADV)<br><br>σ=2%, Y=0.7, Q/ADV=5% → 22 bps</div>
      <div class="fs-note">Y≈0.5–1.0; calibrate per asset class</div>
    </div>
    <div class="fs-item">
      <div class="fs-label">Break-even Alpha</div>
      <div class="fs-formula">Min_Alpha = TC_annual / Sharpe_hurdle<br><br>or: Alpha > TC to survive at all</div>
      <div class="fs-note">Stress-test: what if costs are 2×?</div>
    </div>
  </div>
</div>

<!-- ═══ TURNOVER DRAG MATRIX ═══ -->
<div class="block">
  <div class="sec-eyebrow">§4 <span>/ Turnover × Cost Matrix</span></div>
  <div class="sec-h">Annual Cost Drag — The Numbers That Kill Strategies</div>
  <div class="drag-section">

    <div class="drag-card">
      <div class="drag-title">Annual TC Drag = Turnover × Round-Trip Cost (bps)</div>
      <table class="matrix">
        <thead>
          <tr>
            <th>Turnover</th>
            <th>5 bps RT</th>
            <th>10 bps RT</th>
            <th>20 bps RT</th>
            <th>50 bps RT</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>100% pa</td>
            <td class="td-lo">0.05%</td>
            <td class="td-lo">0.10%</td>
            <td class="td-lo">0.20%</td>
            <td class="td-lo">0.50%</td>
          </tr>
          <tr>
            <td>200% pa</td>
            <td class="td-lo">0.10%</td>
            <td class="td-lo">0.20%</td>
            <td class="td-lo">0.40%</td>
            <td class="td-md">1.00%</td>
          </tr>
          <tr>
            <td>500% pa</td>
            <td class="td-lo">0.25%</td>
            <td class="td-md">0.50%</td>
            <td class="td-md">1.00%</td>
            <td class="td-hi">2.50%</td>
          </tr>
          <tr>
            <td>1000% pa</td>
            <td class="td-md">0.50%</td>
            <td class="td-md">1.00%</td>
            <td class="td-hi">2.00%</td>
            <td class="td-ex">5.00%</td>
          </tr>
          <tr>
            <td>2000% pa</td>
            <td class="td-md">1.00%</td>
            <td class="td-hi">2.00%</td>
            <td class="td-ex">4.00%</td>
            <td class="td-ex">10.0%</td>
          </tr>
          <tr>
            <td>5000% pa</td>
            <td class="td-hi">2.50%</td>
            <td class="td-ex">5.00%</td>
            <td class="td-ex">10.0%</td>
            <td class="td-ex">25.0%</td>
          </tr>
        </tbody>
      </table>
      <div style="font-size:11px;color:var(--muted);margin-top:12px;font-family:var(--mono)">
        🟢 Survivable &nbsp;|&nbsp; 🟡 Significant drag &nbsp;|&nbsp; 🔴 Requires very high gross alpha &nbsp;|&nbsp; <span style="color:var(--red);font-weight:700">RED</span> = Strategy-killing
      </div>
    </div>

    <div class="drag-card">
      <div class="drag-title">Break-even Gross Alpha Required (at Sharpe target = 1.0)</div>
      <div class="be-chart" style="margin-top:12px;">
        <div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:8px;letter-spacing:1px;">MIN GROSS ALPHA NEEDED TO SURVIVE COSTS</div>
        <div class="be-row">
          <div class="be-label">Low-freq · 100% TO<br>10 bps RT</div>
          <div class="be-track"><div class="be-fill" style="width:8%;background:var(--teal);"><span>0.1%</span></div></div>
          <div class="be-note" style="color:var(--teal)">Easy</div>
        </div>
        <div class="be-row">
          <div class="be-label">Mid-freq · 500% TO<br>20 bps RT</div>
          <div class="be-track"><div class="be-fill" style="width:33%;background:var(--gold);"><span>1.0%</span></div></div>
          <div class="be-note" style="color:var(--gold)">Moderate</div>
        </div>
        <div class="be-row">
          <div class="be-label">High-freq · 2000% TO<br>20 bps RT</div>
          <div class="be-track"><div class="be-fill" style="width:67%;background:var(--orange);"><span>4.0%</span></div></div>
          <div class="be-note" style="color:var(--orange)">Hard</div>
        </div>
        <div class="be-row">
          <div class="be-label">Ultra-HF · 5000% TO<br>50 bps RT</div>
          <div class="be-track"><div class="be-fill" style="width:100%;background:var(--red);"><span>25.0%</span></div></div>
          <div class="be-note" style="color:var(--red)">Extreme</div>
        </div>
      </div>
      <div style="font-size:11px;color:var(--muted);margin-top:20px;line-height:1.7;">
        <strong style="color:var(--ink)">Key insight:</strong> As turnover increases, the bar for gross alpha rises non-linearly. Most "signal research" at high frequency underestimates this bar by 3–10×.
      </div>
    </div>

  </div>
</div>

<!-- ═══ ASSET CLASS BENCHMARKS ═══ -->
<div class="block">
  <div class="sec-eyebrow">§5 <span>/ Asset Class Cost Benchmarks</span></div>
  <div class="sec-h">Typical Transaction Costs by Asset Class</div>
  <div class="asset-wrap">
    <table class="asset-table">
      <thead>
        <tr>
          <th>Asset Class</th>
          <th>Spread (bps)</th>
          <th>Impact per 5% ADV</th>
          <th>Commission</th>
          <th>Liquidity</th>
          <th>Total RT Cost Est.</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="asset-name">US Large-Cap Equity</span></td>
          <td><span class="badge-sm b-lo">1–3 bps</span></td>
          <td><span class="badge-sm b-lo">8–15 bps</span></td>
          <td>0.5–1 bps</td>
          <td>
            <div class="dot-gauge">
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
            </div>
          </td>
          <td><strong style="color:var(--teal)">10–20 bps RT</strong></td>
          <td style="font-size:12px;color:var(--muted)">Most liquid. Spread near tick-constrained.</td>
        </tr>
        <tr>
          <td><span class="asset-name">US Mid-Cap Equity</span></td>
          <td><span class="badge-sm b-md">5–15 bps</span></td>
          <td><span class="badge-sm b-md">15–30 bps</span></td>
          <td>1–2 bps</td>
          <td>
            <div class="dot-gauge">
              <div class="dg fill" style="background:var(--gold)"></div>
              <div class="dg fill" style="background:var(--gold)"></div>
              <div class="dg fill" style="background:var(--gold)"></div>
              <div class="dg empty"></div><div class="dg empty"></div>
            </div>
          </td>
          <td><strong style="color:var(--gold)">25–50 bps RT</strong></td>
          <td style="font-size:12px;color:var(--muted)">Spread widens sharply in stress.</td>
        </tr>
        <tr>
          <td><span class="asset-name">US Small-Cap Equity</span></td>
          <td><span class="badge-sm b-hi">20–100 bps</span></td>
          <td><span class="badge-sm b-hi">30–100+ bps</span></td>
          <td>2–5 bps</td>
          <td>
            <div class="dot-gauge">
              <div class="dg fill" style="background:var(--red)"></div>
              <div class="dg fill" style="background:var(--red)"></div>
              <div class="dg empty"></div><div class="dg empty"></div><div class="dg empty"></div>
            </div>
          </td>
          <td><strong style="color:var(--red)">60–200+ bps RT</strong></td>
          <td style="font-size:12px;color:var(--muted)">Capacity severely constrained. Impact dominates.</td>
        </tr>
        <tr>
          <td><span class="asset-name">S&P 500 Futures (ES)</span></td>
          <td><span class="badge-sm b-lo">0.5–1 bps</span></td>
          <td><span class="badge-sm b-lo">3–8 bps</span></td>
          <td>0.2–0.5 bps</td>
          <td>
            <div class="dot-gauge">
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
            </div>
          </td>
          <td><strong style="color:var(--teal)">5–10 bps RT</strong></td>
          <td style="font-size:12px;color:var(--muted)">Deepest single-contract market globally.</td>
        </tr>
        <tr>
          <td><span class="asset-name">FX Spot (Majors)</span></td>
          <td><span class="badge-sm b-lo">0.5–2 bps</span></td>
          <td><span class="badge-sm b-lo">2–5 bps</span></td>
          <td>0.1–0.5 bps</td>
          <td>
            <div class="dot-gauge">
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
              <div class="dg fill" style="background:var(--teal)"></div>
            </div>
          </td>
          <td><strong style="color:var(--teal)">3–8 bps RT</strong></td>
          <td style="font-size:12px;color:var(--muted)">$7.5T daily volume. Near-zero impact at most sizes.</td>
        </tr>
        <tr>
          <td><span class="asset-name">US Corporate Bonds</span></td>
          <td><span class="badge-sm b-hi">30–200 bps</span></td>
          <td><span class="badge-sm b-hi">50–300 bps</span></td>
          <td>variable</td>
          <td>
            <div class="dot-gauge">
              <div class="dg fill" style="background:var(--red)"></div>
              <div class="dg empty"></div><div class="dg empty"></div><div class="dg empty"></div><div class="dg empty"></div>
            </div>
          </td>
          <td><strong style="color:var(--red)">100–500 bps RT</strong></td>
          <td style="font-size:12px;color:var(--muted)">OTC; dealer-intermediated. No central limit order book.</td>
        </tr>
        <tr>
          <td><span class="asset-name">Crypto (BTC/ETH)</span></td>
          <td><span class="badge-sm b-var">5–20 bps</span></td>
          <td><span class="badge-sm b-var">10–80 bps</span></td>
          <td>5–15 bps</td>
          <td>
            <div class="dot-gauge">
              <div class="dg fill" style="background:var(--purple)"></div>
              <div class="dg fill" style="background:var(--purple)"></div>
              <div class="dg empty"></div><div class="dg empty"></div><div class="dg empty"></div>
            </div>
          </td>
          <td><strong style="color:var(--purple)">30–120 bps RT</strong></td>
          <td style="font-size:12px;color:var(--muted)">Fragmented venues; no consolidated tape. Highly variable.</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ═══ IMPACT MECHANICS ═══ -->
<div class="block">
  <div class="sec-eyebrow">§6 <span>/ Impact Mechanics</span></div>
  <div class="sec-h">Permanent vs Temporary Impact — The Critical Distinction</div>
  <div class="impact-grid">
    <div class="impact-card">
      <div class="ic-title">Permanent Impact</div>
      <div class="ic-sub">Price never recovers — fundamental info</div>
      <div class="ic-formula">ΔP_permanent = g(rate) × |size|<br><br>Persists after trade completes.<br>Reflects information content of order.</div>
      <div class="ic-points">
        <div class="ic-point"><div class="ic-dot" style="background:var(--red)"></div>Your trade reveals private information → market adjusts price permanently.</div>
        <div class="ic-point"><div class="ic-dot" style="background:var(--red)"></div>Cannot be reduced by slower execution — the info leaks regardless.</div>
        <div class="ic-point"><div class="ic-dot" style="background:var(--red)"></div>Roughly 40–60% of total impact for informed institutional orders.</div>
      </div>
    </div>
    <div class="impact-card">
      <div class="ic-title">Temporary Impact</div>
      <div class="ic-sub">Price reverts after trade — pure liquidity cost</div>
      <div class="ic-formula">ΔP_temp = h(rate) × sgn(size)<br><br>Reverts as book refills post-execution.<br>Decay: power law, τ^β, β ≈ 0.25–0.5</div>
      <div class="ic-points">
        <div class="ic-point"><div class="ic-dot" style="background:var(--teal)"></div>Caused by consuming book depth. Market makers refill over time.</div>
        <div class="ic-point"><div class="ic-dot" style="background:var(--teal)"></div>Reduced by slowing execution — VWAP/TWAP allows refilling between child orders.</div>
        <div class="ic-point"><div class="ic-dot" style="background:var(--teal)"></div>Typically 40–60% of total impact. The "manageable" component.</div>
      </div>
    </div>
    <div class="impact-card">
      <div class="ic-title">Almgren-Chriss Framework</div>
      <div class="ic-sub">Optimal trade scheduling under impact</div>
      <div class="ic-formula">Total Cost = ½ × g(v)|X| + h(v)|X|<br><br>Optimal: balance impact cost<br>vs timing risk (variance × aversion)</div>
      <div class="ic-points">
        <div class="ic-point"><div class="ic-dot" style="background:var(--gold)"></div>Slow down to reduce impact, but increase risk of adverse price drift.</div>
        <div class="ic-point"><div class="ic-dot" style="background:var(--gold)"></div>Risk-averse traders execute faster. Risk-neutral traders execute more patiently.</div>
        <div class="ic-point"><div class="ic-dot" style="background:var(--gold)"></div>Foundation of all institutional VWAP/TWAP scheduling algorithms.</div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ REGIME MULTIPLIERS ═══ -->
<div class="block">
  <div class="sec-eyebrow">§7 <span>/ Regime-Conditional Costs</span></div>
  <div class="sec-h">Transaction Costs Are Not Constant — They Spike in Stress</div>
  <div class="regime-grid">
    <div class="reg-cell">
      <div class="reg-name">Normal Market</div>
      <div class="reg-mult rm-normal">1.0×</div>
      <div class="reg-desc">Baseline. Tight spreads. Full depth. Market makers competitive. Square root model calibrated here — most backtests assume this regime always holds.</div>
    </div>
    <div class="reg-cell">
      <div class="reg-name">Elevated Vol (VIX 25–40)</div>
      <div class="reg-mult rm-stress">2–3×</div>
      <div class="reg-desc">Spreads widen. Depth thins. Borrow rates rise for crowded shorts. A backtest using normal-period costs understates costs in every stressed month.</div>
    </div>
    <div class="reg-cell">
      <div class="reg-name">Market Stress (VIX 40+)</div>
      <div class="reg-mult rm-crisis">4–6×</div>
      <div class="reg-desc">2008, Mar 2020. Mid-cap spreads 4–6× normal. Borrow rates spike. Impact multiplier collapses capacity estimates. OTC bonds become untradeable.</div>
    </div>
    <div class="reg-cell">
      <div class="reg-name">Flash Crash / Halt</div>
      <div class="reg-mult rm-crash">10×+</div>
      <div class="reg-desc">Depth evaporates. Stops fill at gap prices far from trigger. Any fixed cost assumption is fiction. The strategy's drawdown in these events is far worse than backtest shows.</div>
    </div>
  </div>
</div>

<!-- ═══ RED FLAGS ═══ -->
<div class="block">
  <div class="sec-eyebrow">§8 <span>/ Red Flags</span></div>
  <div class="sec-h">If Your Backtest Does Any of These — Your P&amp;L Is Fictional</div>
  <div class="flags-grid">
    <div class="flag">
      <div class="flag-name">Fixed cost regardless of order size</div>
      <div class="flag-body">Same 10 bps for 100 shares and 500,000 shares. Market impact scales with order size via square root law. Small orders subsidise large ones in your model.</div>
      <div class="flag-fix">✓ Fix: Impact = σ × Y × √(Q/ADV) per trade</div>
    </div>
    <div class="flag">
      <div class="flag-name">Using current spreads for 2008 data</div>
      <div class="flag-body">2008 spreads were 5× 2024 levels. Using tight modern spreads for crisis simulation makes every stressed period look trivially cheap. Drawdowns appear smaller than reality.</div>
      <div class="flag-fix">✓ Fix: Use time-series of historical spreads from TAQ/Bloomberg</div>
    </div>
    <div class="flag">
      <div class="flag-name">No borrow cost on short book</div>
      <div class="flag-body">Short side has borrow cost (0.25–50%+ pa) on top of the same spread and impact as longs. Strategies that systematically short small-caps or squeeze candidates are ignoring the largest single cost.</div>
      <div class="flag-fix">✓ Fix: Add borrow_rate × |short_notional| / 252 per day</div>
    </div>
    <div class="flag">
      <div class="flag-name">Infinite capacity assumed</div>
      <div class="flag-body">Strategy fills 100% of intended size at every rebalance regardless of position vs available volume. At institutional scale, even 5% of ADV per order is a material impact assumption.</div>
      <div class="flag-fix">✓ Fix: Cap fills at participation rate × bar volume (5–20%)</div>
    </div>
    <div class="flag">
      <div class="flag-name">SR collapses at 2× modelled costs</div>
      <div class="flag-body">If doubling the cost assumption halves your Sharpe, the strategy is cost-fragile. Real costs frequently exceed model costs in live trading due to model miscalibration and regime shifts.</div>
      <div class="flag-fix">✓ Fix: Stress-test at 1×, 2×, 5× modelled costs. Require SR ≥ 0.5 at 2×.</div>
    </div>
    <div class="flag">
      <div class="flag-name">Opportunity cost ignored</div>
      <div class="flag-body">Shares you intended to trade but couldn't (liquidity, price moved) represent alpha leakage. Implementation Shortfall includes this. Treating unexecuted orders as zero-cost is a systematic positive bias.</div>
      <div class="flag-fix">✓ Fix: Use IS framework; penalise unexecuted quantity at decision drift</div>
    </div>
  </div>
</div>

<!-- ═══ CHECKLIST ═══ -->
<div class="block">
  <div class="sec-eyebrow">§9 <span>/ Implementation Checklist</span></div>
  <div class="sec-h">Before Trusting Any Cost-Inclusive Backtest</div>
  <div class="cl-grid">
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">All 6 components modelled: spread, impact, commission, borrow, taxes, opportunity cost.</div></div>
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">Spread is time-varying (historical data). Not a fixed constant across all periods.</div></div>
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">Market impact uses square root law with asset-specific Y. Capacity-tested at 2×, 5× AUM.</div></div>
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">Borrow costs applied to short book. Hard-to-borrow names flagged and excluded or penalised.</div></div>
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">Regime-varying costs: 2008 and 2020 stress periods use 3–5× normal spread estimates.</div></div>
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">Annual TC drag computed: Turnover × RT_Cost. Compared against gross alpha directly.</div></div>
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">Partial fills modelled. Max fill = participation rate × bar volume. Unfilled re-queued.</div></div>
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">Cost sensitivity test: Sharpe at 1×, 2×, 5× modelled costs. Strategy must survive 2×.</div></div>
    <div class="cl-item"><div class="cl-box">✓</div><div class="cl-text">IS decomposed into spread, impact, and opportunity cost. Alpha is measured from decision price.</div></div>
  </div>
</div>

</main>

<footer>Transaction Costs · Quant Finance Theory · Visual Reference</footer>

</body>
</html>

# Convert to PDF
HTML(string=html_content).write_pdf('transaction_costs.pdf')
print("PDF created successfully!")

from weasyprint import HTML

HTML('transaction_costs.html').write_pdf('transaction_costs.pdf')
print("PDF created successfully!")