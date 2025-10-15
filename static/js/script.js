// script.js
// Beginner-friendly: DOM updates + calls to fakeApi.*
// Change to real fetch endpoints later (see notes below).

document.addEventListener('DOMContentLoaded', () => {
  init();
});

const state = {
  page: 1,
  perPage: 8,
  totalOrganisms: 0,
  search: '',
  filter: ''
};

async function init(){
  await loadStats();
  await loadRecent();
  await loadOrganisms();
  attachListeners();
}

function attachListeners(){
  document.getElementById('searchInput').addEventListener('input', debounce((e)=>{
    state.search = e.target.value.trim();
    state.page = 1;
    loadOrganisms();
  }, 300));

  document.getElementById('filterSelect').addEventListener('change', (e)=>{
    state.filter = e.target.value;
    state.page = 1;
    loadOrganisms();
  });

  document.getElementById('uploadBtn').addEventListener('click', async ()=>{
    const f = document.getElementById('fileInput').files[0];
    if(!f){ alert('Please choose an image file first.'); return; }
    const btn = document.getElementById('uploadBtn');
    btn.disabled = true; btn.textContent = 'Uploading...';
    try {
      const res = await fakeApi.uploadImage(f);
      alert(res.message + '\nDetected: ' + res.analysis.organismsDetected + ' organisms\nAvg conf: ' + res.analysis.avgConfidence + '%');
      // After upload we can refresh lists
      loadRecent();
      loadOrganisms();
    } catch(err){
      console.error(err);
      alert('Upload failed (mock).');
    } finally {
      btn.disabled = false; btn.textContent = 'Upload';
    }
  });

  document.getElementById('captureBtn').addEventListener('click', async ()=>{
    const btn = document.getElementById('captureBtn');
    btn.disabled = true; btn.textContent = 'Capturing...';
    try {
      const sample = await fakeApi.captureSample();
      alert('Captured sample: ' + sample.title + ' — detected ' + sample.count + ' organisms (mock).');
      // add to recent list by reloading
      loadRecent();
    } catch(err){
      console.error(err);
    } finally {
      btn.disabled = false; btn.textContent = 'Start Capture';
    }
  });

  document.getElementById('learnBtn').addEventListener('click', ()=>{
    window.scrollTo({top: document.querySelector('.recent-samples').offsetTop - 20, behavior: 'smooth'});
  });
}

async function loadStats(){
  const s = await fakeApi.getStats();
  document.getElementById('detected').textContent = s.detected.toLocaleString();
  document.getElementById('species').textContent = s.species.toLocaleString();
  document.getElementById('confidence').textContent = s.confidence + '%';
  document.getElementById('totalSamples').textContent = s.totalSamples;
  document.getElementById('organismsFound').textContent = s.organismsFound;
  document.getElementById('speciesIdentified').textContent = s.speciesIdentified;
}

async function loadRecent(){
  const recent = await fakeApi.getRecentSamples();
  const ul = document.getElementById('recentList');
  ul.innerHTML = '';
  recent.forEach(r=>{
    const li = document.createElement('li');
    li.innerHTML = `
      <div class="recent-thumb"><img src="${r.thumb}" alt="" style="width:100%;height:100%;object-fit:cover;border-radius:6px"/></div>
      <div>
        <div style="font-weight:600">${r.title}</div>
        <div style="font-size:13px;color:var(--muted)">${r.when} · ${r.count} organisms · ${r.conf}%</div>
      </div>
    `;
    ul.appendChild(li);
  });
}

async function loadOrganisms(page = state.page){
  const res = await fakeApi.getOrganisms({
    page,
    perPage: state.perPage,
    search: state.search,
    filter: state.filter
  });
  state.totalOrganisms = res.total;
  state.page = page;
  renderOrganismsTable(res.items);
  renderPagination(res.total, page, state.perPage);
}

function renderOrganismsTable(items){
  const tbody = document.getElementById('orgTableBody');
  tbody.innerHTML = '';
  if(items.length === 0){
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--muted)">No organisms found</td></tr>`;
    return;
  }
  for(const o of items){
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><img class="thumb" src="${o.thumbnail}" alt="${o.name}"/></td>
      <td><strong>${o.name}</strong></td>
      <td>${o.count}</td>
      <td>${o.size}</td>
      <td>${o.confidence}%</td>
      <td>${o.category}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderPagination(total, page, perPage){
  const container = document.getElementById('pagination');
  container.innerHTML = '';
  const pages = Math.max(1, Math.ceil(total / perPage));
  const createBtn = (p, text = p) => {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.className = 'page-btn' + (p === page ? ' active' : '');
    btn.addEventListener('click', ()=> loadOrganisms(p));
    return btn;
  };

  // show prev, page numbers, next
  if(page > 1) container.appendChild(createBtn(page-1, 'Prev'));
  const start = Math.max(1, page - 2);
  const end = Math.min(pages, page + 2);
  for(let p = start; p <= end; p++){
    container.appendChild(createBtn(p));
  }
  if(page < pages) container.appendChild(createBtn(page+1, 'Next'));
}

/* simple debounce */
function debounce(fn, ms=200){
  let t;
  return function(...args){
    clearTimeout(t);
    t = setTimeout(()=> fn.apply(this, args), ms);
  };
}
