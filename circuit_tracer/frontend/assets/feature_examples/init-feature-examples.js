window.initFeatureExamples = function({containerSel, showLogits=true, showExamples=true, hideStaleOutputs=false}){
  var visState = {
    isDev: 0,
    showLogits,
    showExamples,
    hideStaleOutputs,

    activeToken: null,
    feature: null,
    featureIndex: -1,

    chartRowTop: 16,
    chartRowHeight: 82,
  }

  // set up dom and render fns
  var sel = containerSel.html('').append('div.feature-examples')
  if (visState.showLogits) sel.append('div.feature-example-logits')
  if (visState.showExamples) sel.append('div.feature-example-list')
  var renderAll = util.initRenderAll(['feature'])

  
  if (visState.showLogits) window.initFeatureExamplesLogits({renderAll, visState, sel})
  if (visState.showExamples) window.initFeatureExamplesList({renderAll, visState, sel})

  return {loadFeature, renderFeature}

  async function renderFeature(scan, featureIndex){
    if (visState.hideStaleOutputs) sel.classed('is-stale-output', 1)
    if (featureIndex == visState.featureIndex) return
    // load feature data and exit early if featureIndex has changed
    visState.featureIndex = featureIndex
    var feature = await loadFeature(scan, featureIndex)
    if (feature.featureIndex == visState.featureIndex){
      visState.feature = feature
      renderAll.feature()
      if (visState.hideStaleOutputs) sel.classed('is-stale-output', 0)
    }

    return feature
  }

  async function loadFeature(scan, featureIndex){
    try {
      if (scan.startsWith('./')) {
        var feature = await  util.getFile(`${scan}/${featureIndex}.json`)
      } else {
        var feature = await  util.getFile(`./features/${scan}/${featureIndex}.json`)
      }
    } catch {
      var feature = {isDead: true, statistics: {}}
    }

    if (feature.act_min === undefined) {
      feature.act_min = 0
      feature.act_max = 1.4
    }

    feature.colorScale = d3.scaleSequential(d3.interpolateOranges)
      .domain([feature.act_min, feature.act_max]).clamp(1)

    feature.featureIndex = featureIndex
    feature.scan = scan

    return feature
  }
}

window.init?.()
