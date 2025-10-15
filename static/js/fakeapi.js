// fakeApi.js
// Simple mocked API for frontend development.
// Replace functions with real fetch() calls later.

(function(global){
  const placeholder = 'data:image/svg+xml;utf8,'
    + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80"><rect width="100%" height="100%" fill="%23cfeeff"/><text x="8" y="45" font-size="14" fill="%23005599">Sample</text></svg>');

  const MOCK_STATS = {
    detected: 12847,
    species: 2156,
    confidence: 94.2,
    totalSamples: 1247,
    organismsFound: 8932,
    speciesIdentified: 156
  };

  const recentSamples = [
    { id:1, title:"Plankton Sample A", when:"2 hours ago", count:23, conf:96.8, thumb:placeholder },
    { id:2, title:"Coastal Water B", when:"5 hours ago", count:31, conf:91.4, thumb:placeholder },
    { id:3, title:"Deep Sea Sample", when:"1 day ago", count:18, conf:89.2, thumb:placeholder }
  ];

  // generate a list of organisms
  const categories = ["Phytoplankton","Zooplankton","Crustacean","Unidentified"];
  const organisms = Array.from({length:30}, (_,i) => {
    const name = ["Coscinodiscus sp.","Copepod sp.","Diatom A","Navicula sp.","Tintinnid sp.","Dino X"][i%6] + " " + (i+1);
    return {
      id: i+1,
      name,
      thumbnail: placeholder,
      count: Math.floor(Math.random()*60)+1,
      size: `${Math.floor(Math.random()*300)+20} µm`,
      confidence: +(70 + Math.random()*30).toFixed(1),
      category: categories[i % categories.length]
    };
  });

  function delay(result, ms=350){
    return new Promise(res => setTimeout(()=>res(result), ms));
  }

  const API = {
    getStats(){ return delay(Object.assign({}, MOCK_STATS)) },
    getRecentSamples(){ return delay(recentSamples.slice(0,5)) },
    /**
     * params: {page, perPage, search, filter}
     */
    getOrganisms({page=1, perPage=8, search='', filter=''} = {}){
      let list = organisms.slice();
      if(search){
        const s = search.toLowerCase();
        list = list.filter(o => o.name.toLowerCase().includes(s));
      }
      if(filter){
        list = list.filter(o => o.category === filter);
      }
      const total = list.length;
      const start = (page-1)*perPage;
      const items = list.slice(start, start+perPage);
      return delay({items, total});
    },
    uploadImage(file){
      // emulate upload + analysis
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            success: true,
            message: "File received and analyzed (mock).",
            analysis: {
              organismsDetected: Math.floor(Math.random()*20)+1,
              topSpecies: "Diatom A",
              avgConfidence: +(80 + Math.random()*15).toFixed(1)
            }
          });
        }, 700);
      });
    },
    captureSample(){
      // emulates live capture -> returns a sample object
      return delay({
        id: Date.now(),
        title: "Live Capture " + new Date().toLocaleTimeString(),
        when: "just now",
        count: Math.floor(Math.random()*40)+1,
        conf: +(75 + Math.random()*20).toFixed(1),
        thumb: placeholder
      }, 800);
    }
  };

  global.fakeApi = API;
})(window);
