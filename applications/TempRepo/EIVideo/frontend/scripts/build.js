process.env.NODE_ENV = 'production'

const {join} = require('path')
const {build} = require('vite')
const chalk = require('chalk')
const {build: electronBuilder} = require('electron-builder')
const {stat, remove, writeFile} = require('fs-extra')
const {rollup} = require('rollup')
const {loadRollupConfig} = require('./util')

/**
 * Generate the distribution version of package json
 */
async function generatePackageJson() {
  const original = require('../package.json')
  const result = {
    name: original.name,
    author: original.author,
    version: original.version,
    license: original.license,
    description: original.description,
    main: './index.js',
    dependencies: Object.entries(original.dependencies).filter(([name, version]) => original.external.indexOf(name) !== -1).reduce((object, entry) => ({
      ...object,
      [entry[0]]: entry[1]
    }), {})
  }
  await writeFile('dist/package.json', JSON.stringify(result))
}

/**
 * Print the rollup output
 * @param {import('rollup').RollupOutput} output
 */
async function printOutput({output}) {
  for (const chunk of output) {
    if (chunk.type === 'chunk') {
      const filepath = join('dist', chunk.fileName)
      const {size} = await stat(join(__dirname, '..', filepath))
      console.log(
        `${chalk.gray('[write]')} ${chalk.cyan(filepath)}  ${(
          size / 1024
        ).toFixed(2)}kb`
      )
    }
  }
}

/**
 * Use rollup to build main process
 * @param {import('rollup').RollupOptions} config
 */
async function buildMain(config) {
  const input = {
    index: join(__dirname, '../src/main/index.ts')
  }

  const bundle = await rollup({
    ...config,
    input
  })
  if (!config.output) {
    throw new Error('Unexpected rollup config to build!')
  }

  await printOutput(await bundle.write(config.output[0]))
}

/**
 * Use vite to build renderer process
 */
function buildRenderer() {
  const config = require('./vite.config')

  console.log(chalk.bold.underline('Build renderer process'))

  return build({
    ...config,
    mode: process.env.NODE_ENV
  })
}

/**
 * Use electron builder to build your app to installer, zip, or etc.
 *
 * @param {import('electron-builder').Configuration} config The electron builder config
 * @param {boolean} dir Use dir mode to build
 */
async function buildElectron(config, dir) {
  console.log(chalk.bold.underline('Build electron'))
  const start = Date.now()
  const files = await electronBuilder({publish: 'never', config, dir})

  for (const file of files) {
    const fstat = await stat(file)
    console.log(
      `${chalk.gray('[write]')} ${chalk.yellow(file)} ${(
        fstat.size /
        1024 /
        1024
      ).toFixed(2)}mb`
    )
  }

  console.log(
    `Build completed in ${((Date.now() - start) / 1000).toFixed(2)}s.`
  )
}

async function start() {
  /**
   * Load electron-builder Configuration
   */
  function loadElectronBuilderConfig() {
    switch (process.env.BUILD_TARGET) {
      case 'production':
        return require('./build.config')
      default:
        return require('./build.lite.config')
    }
  }

  const [mainConfig] = await loadRollupConfig()

  await remove(join(__dirname, '../dist'))

  console.log(chalk.bold.underline('Build main process & preload'))
  const startTime = Date.now()
  await buildMain(mainConfig)
  console.log(
    `Build completed in ${((Date.now() - startTime) / 1000).toFixed(2)}s.\n`
  )
  await buildRenderer()

  console.log()
  if (process.env.BUILD_TARGET) {
    const config = loadElectronBuilderConfig()
    const dir = process.env.BUILD_TARGET === 'dir'
    await generatePackageJson()
    await buildElectron(config, dir)
  }
}

start().catch((e) => {
  console.error(chalk.red(e.toString()))
  process.exit(1)
})
