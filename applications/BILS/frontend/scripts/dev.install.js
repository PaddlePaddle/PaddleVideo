const got = require('got')
const {createWriteStream} = require('fs-extra')
const {pipeline} = require('stream')
const extract = require('extract-zip')
const {join} = require('path')
const {unlinkSync} = require('fs')

async function getLatest() {
  const result = await got('https://api.github.com/repos/vuejs/vue-devtools/releases?per_page=10', {
    headers: {
      accept: 'application/vnd.github.v3+json'
    },
    json: true
  })
  let founded
  for (const release of result.body) {
    const xpi = release.assets.find(a => a.name.endsWith('.xpi'))
    if (xpi) {
      founded = release
      break
    }
  }
  if (founded) {
    const url = founded.assets.find(a => a.name.endsWith('.xpi')).browser_download_url
    const stream = got.stream(url)
    const zipDest = join(__dirname, '../extensions.zip')
    await new Promise((resolve, reject) => {
      pipeline(stream, createWriteStream(zipDest), (e) => {
        if (e) reject(e)
        else resolve(undefined)
      })
    })
    const dir = join(__dirname, '../extensions')
    await new Promise((resolve, reject) => {
      extract(zipDest, {dir}, (err) => {
        if (err) reject(err)
        else resolve(undefined)
      })
    })
    unlinkSync(zipDest)
  }
}

getLatest()
