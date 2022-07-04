import {
  createSemanticDiagnosticsBuilderProgram,
  createWatchCompilerHost,
  createWatchProgram,
  DiagnosticCategory,
  formatDiagnosticsWithColorAndContext,
  sys
} from 'typescript'

/**
 * @param {number | void} timeout
 */
function createDeferred(timeout) {
  let promise
  let resolve = () => {
  }

  if (timeout) {
    promise = Promise.race([
      // eslint-disable-next-line promise/param-names
      new Promise((r) => setTimeout(r, timeout, true)),
      // @ts-ignore
      // eslint-disable-next-line promise/param-names
      new Promise((r) => (resolve = r))
    ])
  } else {
    // @ts-ignore
    // eslint-disable-next-line promise/param-names
    promise = new Promise((r) => (resolve = r))
  }

  return {promise, resolve}
}

const FILE_CHANGE_DETECTED = 6032
const FOUND_1_ERROR_WATCHING_FOR_FILE_CHANGES = 6193
const FOUND_N_ERRORS_WATCHING_FOR_FILE_CHANGES = 6194

/**
 * Typescript watch program helper to sync Typescript watch status with Rollup hooks.
 */
export class WatchProgramHelper {
  _startDeferred = null;
  _finishDeferred = null;

  watch(timeout = 1000) {
    // Race watcher start promise against a timeout in case Typescript and Rollup change detection is not in sync.
    this._startDeferred = createDeferred(timeout)
    this._finishDeferred = createDeferred()
  }

  /**
   * @param {import('typescript').Diagnostic} diagnostic
   */
  handleStatus(diagnostic) {
    // Fullfil deferred promises by Typescript diagnostic message codes.
    if (diagnostic.category === DiagnosticCategory.Message) {
      switch (diagnostic.code) {
        case FILE_CHANGE_DETECTED:
          this.resolveStart()
          break

        case FOUND_1_ERROR_WATCHING_FOR_FILE_CHANGES:
        case FOUND_N_ERRORS_WATCHING_FOR_FILE_CHANGES:
          this.resolveFinish()
          break

        default:
      }
    }
  }

  resolveStart() {
    if (this._startDeferred) {
      this._startDeferred.resolve(false)
      this._startDeferred = null
    }
  }

  resolveFinish() {
    if (this._finishDeferred) {
      this._finishDeferred.resolve(false)
      this._finishDeferred = null
    }
  }

  async wait() {
    if (this._startDeferred) {
      const timeout = await this._startDeferred.promise

      // If there is no file change detected by Typescript skip deferred promises.
      if (timeout) {
        this._startDeferred = null
        this._finishDeferred = null
      }

      if (this._finishDeferred) {
        await this._finishDeferred.promise
      }
    }
  }
}

/**
 * Create a typecheck only typescript plugin
 * @param {{tsconfig?: string[]; tsconfigOverride?: import('typescript').CompilerOptions; wait?: boolean }} options
 * @returns {import('rollup').Plugin}
 */
const create = ({tsconfig, tsconfigOverride, wait} = {}) => {
  const configPath = tsconfig
  if (!configPath) {
    throw new Error("Could not find a valid 'tsconfig.json'.")
  }
  const createProgram = createSemanticDiagnosticsBuilderProgram

  wait = wait ?? true

  /**
   * @type {import('typescript').FormatDiagnosticsHost}
   */
  const formatHost = {
    getCanonicalFileName: path => path,
    getCurrentDirectory: sys.getCurrentDirectory,
    getNewLine: () => sys.newLine
  }

  /**
   * @type {import('typescript').WatchOfConfigFile<any>[]}
   */
  let programs

  const watcher = new WatchProgramHelper()

  /**
   * @type {import('typescript').Diagnostic[]}
   */
  const diagnostics = []

  /**
   * @type {import('rollup').Plugin}
   */
  const plugin = {
    name: 'typescript:checker',
    buildStart() {
      if (!programs) {
        programs = configPath.map((c) => createWatchProgram(createWatchCompilerHost(
          c,
          tsconfigOverride || {noEmit: true, noEmitOnError: false},
          sys,
          createProgram,
          (diagnostic) => {
            diagnostics.push(diagnostic)
          },
          (diagnostic) => watcher.handleStatus(diagnostic)
        )))
      }
    },
    async load(id) {
      if (!id.endsWith('.ts')) {
        return null
      }
      const promise = watcher.wait()
      if (wait) {
        await promise
      } else {
        promise.then(() => {
          if (diagnostics.length > 0) {
            console.error(formatDiagnosticsWithColorAndContext(diagnostics.splice(0), formatHost))
          }
        })
      }
    },
    generateBundle() {
      if (wait && diagnostics.length > 0) {
        const count = diagnostics.length
        console.error(formatDiagnosticsWithColorAndContext(diagnostics.splice(0), formatHost))
        this.error(`Fail to compile the project. Found ${count} errors.`)
      }
    },
    watchChange(id) {
      if (!id.endsWith('.ts')) {
        return
      }
      watcher.watch()
    },
    buildEnd() {
      if (!this.meta.watchMode) {
        programs.forEach(p => p.close())
      }
    }
  }

  return plugin
}

export default create
