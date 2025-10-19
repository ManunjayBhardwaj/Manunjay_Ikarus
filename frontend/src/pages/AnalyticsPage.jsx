import React, {useEffect, useState} from 'react'
import {getAnalytics} from '../api'
import {BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell, ResponsiveContainer, Legend} from 'recharts'
import Card from '../components/ui/Card'

const COLORS = ['#4f46e5','#10b981','#f59e0b','#ef4444','#06b6d4','#7c3aed','#06b6d4']

function MetricCard({title, value, hint}){
  return (
    <div style={{padding:16, borderRadius:8, background:'var(--card)', boxShadow:'var(--shadow)', border:'1px solid rgba(0,0,0,0.04)'}}>
      <div style={{fontSize:12, color:'var(--text)'}}>{title}</div>
      <div style={{fontSize:20, fontWeight:700, marginTop:6, color:'var(--text)'}}>{value}</div>
      {hint && <div style={{fontSize:12, color:'var(--muted)', marginTop:8}}>{hint}</div>}
    </div>
  )
}

function sanitizeLabel(s){
  if(!s && s !== 0) return 'Unknown'
  try{
    // remove surrounding brackets and quotes and stray characters
    let t = String(s)
    // common patterns like [', '], quotes, trailing ] or leading [
    t = t.trim()
    // remove leading/trailing square brackets
    t = t.replace(/^\[+/, '').replace(/\]+$/, '')
    // remove leading/trailing quotes
    t = t.replace(/^['"\s]+/, '').replace(/['"\s]+$/, '')
    // collapse repeated punctuation
    t = t.replace(/\s*[,;:\-]+\s*/g, ' - ')
    // trim again
    t = t.trim()
    if(t.length === 0) return 'Unknown'
    return t
  }catch(e){ return String(s) }
}

function downloadCSV(filename, rows){
  if(!rows || !rows.length) return
  const keys = Object.keys(rows[0])
  const csv = [keys.join(',')].concat(rows.map(r => keys.map(k => '"'+String(r[k] ?? '')+'"').join(','))).join('\n')
  const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.style.display='none'; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export default function AnalyticsPage(){
  const [raw, setRaw] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(()=>{
    getAnalytics().then(d=>{
      setRaw(d || {})
      setLoading(false)
    }).catch(e=>{console.error(e); setRaw({}); setLoading(false)})
  },[])

  if(loading) return <div style={{padding:20}}>Loading analytics...</div>

  // Map backend keys to local variables; provide defaults
  const categoryCountsRaw = raw.categoryCounts || []
  const avgPriceRaw = raw.avgPriceByCategory || []
  const brandsRaw = raw.brandTop || []
  const materialsRaw = raw.materials || []

  // sanitize category labels
  const categoryCounts = categoryCountsRaw.map(c => ({...c, category: sanitizeLabel(c.category)}))
  const avgPrice = avgPriceRaw.map(a => ({...a, category: sanitizeLabel(a.category)}))

  // build a map from category -> count
  const countsMap = {}
  categoryCounts.forEach(c => { countsMap[c.category] = (countsMap[c.category]||0) + (c.count||0) })

  // compute weighted overall average price when possible
  let overallAvgPrice = null
  try{
    let totalValue = 0
    let totalCount = 0
    avgPrice.forEach(a => {
      const count = countsMap[a.category] || 0
      if(a.avg_price != null && !isNaN(a.avg_price) && count>0){
        totalValue += (Number(a.avg_price) * count)
        totalCount += count
      }
    })
    if(totalCount>0) overallAvgPrice = totalValue/totalCount
  }catch(e){ overallAvgPrice = null }

  // fallback: simple mean of avgPrice values if weighted not available
  if(overallAvgPrice === null){
    const vals = avgPrice.map(a=> Number(a.avg_price)).filter(v=> !isNaN(v))
    if(vals.length) overallAvgPrice = vals.reduce((s,x)=>s+x,0)/vals.length
  }

  // compute min/median/max of avg prices
  const priceVals = avgPrice.map(a=> Number(a.avg_price)).filter(v=> !isNaN(v)).sort((a,b)=>a-b)
  const minPrice = priceVals.length? priceVals[0]: null
  const maxPrice = priceVals.length? priceVals[priceVals.length-1]: null
  const medianPrice = priceVals.length? (priceVals.length%2? priceVals[(priceVals.length-1)/2] : (priceVals[priceVals.length/2-1]+priceVals[priceVals.length/2])/2) : null

  const totalItems = categoryCounts.reduce((s,c)=> s + (c.count||0), 0)

  // prepare chart data: pick top N categories by count and join avg price if available
  const topCategories = categoryCounts.slice().sort((a,b)=> b.count - a.count).slice(0,8)
  const mergedForChart = topCategories.map(c => ({category: c.category, count: c.count, avg_price: (avgPrice.find(a=>a.category===c.category)||{}).avg_price || 0}))

  const topBrands = brandsRaw.slice(0,8)
  // Aggregate materials: keep top 8, group the rest into 'Other'
  const materialsSorted = (materialsRaw||[]).slice().sort((a,b)=> b.count - a.count)
  const MATERIAL_KEEP = 8
  let materials = []
  if(materialsSorted.length <= MATERIAL_KEEP){
    materials = materialsSorted
  } else {
    const top = materialsSorted.slice(0, MATERIAL_KEEP)
    const others = materialsSorted.slice(MATERIAL_KEEP)
    const otherCount = others.reduce((s, x) => s + (x.count||0), 0)
    materials = top.concat([{material: 'Other', count: otherCount}])
  }

  return (
    <div>
      <header style={{marginBottom:12}}>
        <h1 className="title">Analytics</h1>
        <p className="subtitle">Key dataset metrics and distributions.</p>
      </header>

      <div className="grid" style={{gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))', marginBottom:16}}>
        <Card><MetricCard title="Total Items (est.)" value={totalItems || 0} /></Card>
        <Card><MetricCard title="Total Categories" value={categoryCounts.length || 0} /></Card>
        <Card><MetricCard title="Average Price (all)" value={overallAvgPrice? Math.round(overallAvgPrice*100)/100 : 0} /></Card>
        <Card><MetricCard title="Distinct Materials" value={materials.length || 0} /></Card>
      </div>

      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:12}}>
        <div style={{color:'var(--text)', fontWeight:600}}>Top categories by item count (shows avg price too)</div>
        <div>
          <button className="btn ghost" onClick={()=> downloadCSV('category_summary.csv', mergedForChart)}>Copy CSV</button>
        </div>
      </div>

      <div className="grid" style={{gridTemplateColumns:'2fr 1fr', gap:20}}>
        <Card className="card-lg">
          <h4 style={{marginTop:0}}>Average Price by Top Categories</h4>
          <div style={{height:360}}>
              <ResponsiveContainer>
              <BarChart data={mergedForChart} margin={{right:12, left:0}}>
                <XAxis dataKey="category" tick={{fontSize:12, fill:'var(--muted)'}} />
                <YAxis tick={{fontSize:12, fill:'var(--muted)'}} />
                <Tooltip formatter={(v,name)=> [v, name]} contentStyle={{background:'var(--card)'}} />
                <Bar dataKey="avg_price" fill={COLORS[0]} radius={[6,6,0,0]} name="Avg Price" />
                <Bar dataKey="count" fill={COLORS[1]} radius={[6,6,0,0]} name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{marginTop:8, fontSize:13, color:'var(--muted)'}}>
            Min avg price: {minPrice ?? '0'} • Median avg price: {medianPrice ?? '0'} • Max avg price: {maxPrice ?? '0'}
          </div>
        </Card>

        <div style={{display:'grid', gap:16}}>
          <Card>
            <h4 style={{marginTop:0}}>Material Distribution</h4>
            <div style={{display:'flex', alignItems:'center', gap:12}}>
              <div style={{width:160, height:160}}>
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie data={materials} dataKey="count" nameKey="material" outerRadius={72} innerRadius={30} paddingAngle={2}>
                      {materials.map((entry, idx)=> <Cell key={idx} fill={COLORS[idx%COLORS.length]} />)}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div style={{flex:1}}>
                <ul style={{listStyle:'none', margin:0, padding:0}}>
                  {materials.map((m, idx) => {
                    const total = materials.reduce((s,x)=> s + (x.count||0), 0)
                    const pct = total? Math.round(((m.count||0)/total)*1000)/10 : 0
                    return (
                      <li key={m.material} style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:6}}>
                        <div style={{display:'flex', alignItems:'center', gap:8}}>
                          <div style={{width:12, height:12, background:COLORS[idx%COLORS.length], borderRadius:3}} />
                          <div style={{fontWeight:600, color:'var(--text)'}}>{m.material}</div>
                        </div>
                        <div style={{color:'var(--muted)'}}>{m.count} • {pct}%</div>
                      </li>
                    )
                  })}
                </ul>
              </div>
            </div>
          </Card>

          <Card>
            <h4 style={{marginTop:0}}>Top Brands</h4>
            <ol style={{margin:0, paddingLeft:18}}>
              {topBrands.map(b => <li key={b.brand} style={{marginBottom:6, color:'var(--text)'}}>{b.brand} <span style={{color:'var(--muted)'}}>({b.count})</span></li>)}
            </ol>
          </Card>
        </div>
      </div>

      <div style={{marginTop:20}}>
        <h4>Top Categories (overview)</h4>
        <div style={{display:'flex', flexWrap:'wrap', gap:8}}>
          {topCategories.map(c => (
            <Card key={c.category} style={{padding:'8px 12px', borderRadius:8}}>
              <div style={{fontWeight:600, color:'var(--text)'}}>{c.category}</div>
              <div style={{fontSize:12, color:'var(--muted)'}}>{c.count} items • avg ${Math.round(((avgPrice.find(a=>a.category===c.category)||{}).avg_price||0)*100)/100}</div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
