process.env.NODE_ENV = 'development'

process.once('exit', terminate)
  .once('SIGINT', terminate)

const electron = require('electron')
const {spawn} = require('child_process')
const {join, resolve} = require('path')
const {createServer} = require('vite')
const {createServer: createSocketServer} = require('net')
const chalk = require('chalk')
const {watch} = require('rollup')
const {EOL} = require('os')
const {loadRollupConfig} = require('./util')
const {remove} = require('fs-extra')

let manualRestart = false

/**
 * @type {import('child_process').ChildProcessWithoutNullStreams  | null}
 */
let electronProcess = null

/**
 * The current active dev socket connecting to the electron main process.
 * Currently, this is only used for preloading the preload script
 * @type {import('net').Socket | null}
 */
let devSocket = null

/**
 * The customized devserver communicated with the electron main process.
 * Currently, this is only used for preloading the preload script
 * @type {import('net').Server  | null}
 */
let devServer = null

/**
 * Start electron process and inspect port 5858 with 9222 as debug port.
 */
function startElectron() {
  /** @type {any} */
  const electronPath = electron
  const spawnProcess = spawn(
    electronPath,
    ['--inspect=5858', '--remote-debugging-port=9222', join(__dirname, '../dist/index.js')]
  )

  /**
   * @param {string | Buffer} data
   */
  function electronLog(data) {
    const colorize = (line) => {
      if (line.startsWith('[INFO]')) {
        return chalk.green('[INFO]') + line.substring(6)
      } else if (line.startsWith('[WARN]')) {
        return chalk.yellow('[WARN]') + line.substring(6)
      } else if (line.startsWith('[ERROR]')) {
        return chalk.red('[ERROR]') + line.substring(7)
      }
      return chalk.grey('[CONSOLE] ') + line
    }
    console.log(
      data.toString()
        .split(EOL)
        .filter(s => s.trim() !== '')
        .filter(s => s.indexOf('source: chrome-extension:') === -1)
        .map(colorize).join(EOL)
    )
  }

  spawnProcess.stdout.on('data', electronLog)
  spawnProcess.stderr.on('data', electronLog)
  spawnProcess.on('exit', (_, signal) => {
    if (!manualRestart) {
      // if (!devtoolProcess.killed) {
      //     devtoolProcess.kill(0);
      // }
      if (!signal) { // Manual close
        process.exit(0)
      }
    } else {
      manualRestart = false
    }
  })

  electronProcess = spawnProcess
}

/**
 * Kill and restart electron process
 */
function reloadElectron() {
  if (electronProcess) {
    manualRestart = true
    electronProcess.kill('SIGTERM')
    console.log(`${chalk.cyan('[DEV]')} ${chalk.bold.underline.green('Electron app restarted')}`)
  } else {
    console.log(`${chalk.cyan('[DEV]')} ${chalk.bold.underline.green('Electron app started')}`)
  }
  startElectron()
}

function reloadPreload() {
  if (devSocket) {
    devSocket.write(Buffer.from([0]))
  }
}

/**
 * Start vite dev server for renderer process and listen 8080 port
 */
async function startRenderer() {
  const config = require('./vite.config')

  config.mode = process.env.NODE_ENV

  const server = await createServer(config)
  return server.listen(8080)
}

/**
 * @param {import('rollup').RollupOptions} config
 */
async function loadMainConfig(config) {
  const input = {
    index: join(__dirname, '../src/main/index.dev.ts')
  }

  return {
    ...config,
    input,
    watch: {
      buildDelay: 500
    }
  }
}

/**
 * Main method of this script
 */
async function main() {
  const [mainConfig] = await loadRollupConfig()

  devServer = createSocketServer((sock) => {
    console.log(`${chalk.cyan('[DEV]')} Dev socket connected`)
    devSocket = sock
    sock.on('error', (e) => {
      // @ts-ignore
      if (e.code !== 'ECONNRESET') {
        console.error(e)
      }
    })
  }).listen(3031, () => {
    console.log(`${chalk.cyan('[DEV]')} Dev server listening on 3031`)
  })

  const preloadPrefix = resolve(__dirname, '../src/preload')
  let shouldReloadElectron = true
  let shouldReloadPreload = false
  const config = await loadMainConfig(mainConfig)
  await startRenderer()

  // start watch the main & preload
  watch(config)
    .on('change', (id) => {
      console.log(`${chalk.cyan('[DEV]')} change ${id}`)
      if (id.startsWith(preloadPrefix)) {
        shouldReloadPreload = true
      } else {
        shouldReloadElectron = true
      }
    })
    .on('event', (event) => {
      switch (event.code) {
        case 'END':
          if (shouldReloadElectron || !electronProcess) {
            reloadElectron()
            shouldReloadElectron = false
          } else {
            console.log(`${chalk.cyan('[DEV]')} Skip start/reload electron.`)
          }
          if (shouldReloadPreload) {
            reloadPreload()
            shouldReloadPreload = false
          } else {
            console.log(`${chalk.cyan('[DEV]')} Skip start/reload preload.`)
          }
          break
        case 'BUNDLE_END':
          console.log(`${chalk.cyan('[DEV]')} Bundle ${event.output} ${event.duration + 'ms'}`)
          break
        case 'ERROR':
          console.error(event)
          shouldReloadElectron = false
          break
      }
    })
}

remove(join(__dirname, '../dist')).then(() => main()).catch(e => {
  console.error(e)
  terminate()
  process.exit(1)
})

function terminate() {
  if (electronProcess) {
    electronProcess.kill()
    electronProcess = null
  }
  if (devSocket) {
    devSocket.destroy()
    devSocket = null
  }
  if (devServer) {
    devServer.close()
    devServer = null
  }
}
