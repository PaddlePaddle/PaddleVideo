import {app, BrowserWindow} from 'electron'
import {createWriteStream} from 'fs'
import {join, resolve} from 'path'
import {PassThrough, pipeline, Transform} from 'stream'
import {format} from 'util'

function formatMsg(message: any, options: any[]) {
  return options.length !== 0 ? format(message, options) : format(message)
}

function baseTransform(tag: string) {
  return new Transform({
    transform(c, e, cb) {
      cb(undefined, `[${tag}] [${new Date().toLocaleString()}] ${c}\n`)
    }
  })
}

export interface LoggerFacade {
  log(message: any, ...options: any[]): void;

  warn(message: any, ...options: any[]): void;

  error(message: any, ...options: any[]): void;
}

export class Logger {
  private loggerEntries = {log: baseTransform('INFO'), warn: baseTransform('WARN'), error: baseTransform('ERROR')};
  private output = new PassThrough();
  private logDirectory: string = ''

  constructor() {
    pipeline(this.loggerEntries.log, this.output, () => {
    })
    pipeline(this.loggerEntries.warn, this.output, () => {
    })
    pipeline(this.loggerEntries.error, this.output, () => {
    })

    process.on('uncaughtException', (err) => {
      this.error('Uncaught Exception')
      this.error(err)
    })
    process.on('unhandledRejection', (reason) => {
      this.error('Uncaught Rejection')
      this.error(reason)
    })
    if (process.env.NODE_ENV === 'development') {
      this.output.on('data', (b) => {
        console.log(b.toString())
      })
    }
    app.once('browser-window-created', (event, window) => {
      this.captureWindowLog(window)
    })
  }

  readonly log = (message: any, ...options: any[]) => {
    this.loggerEntries.log.write(formatMsg(message, options))
  }

  readonly warn = (message: any, ...options: any[]) => {
    this.loggerEntries.warn.write(formatMsg(message, options))
  }

  readonly error = (message: any, ...options: any[]) => {
    this.loggerEntries.error.write(formatMsg(message, options))
  }

  /**
   * Initialize log output directory
   * @param directory The directory of the log
   */
  async initialize(directory: string) {
    this.logDirectory = directory
    const mainLog = join(directory, 'main.log')
    const stream = createWriteStream(mainLog, {encoding: 'utf-8', flags: 'w+'})
    this.output.pipe(stream)
    this.log(`Setup main logger to ${mainLog}`)
  }

  /**
   * Capture the window log
   * @param window The browser window
   * @param name The name alias of the window. Use window.webContents.id by default
   */
  captureWindowLog(window: BrowserWindow, name?: string) {
    name = name ?? window.webContents.id.toString()
    if (!this.logDirectory) {
      this.warn(`Cannot capture window log for window ${name}. Please initialize the logger to set logger directory!`)
      return
    }
    const loggerPath = resolve(this.logDirectory, `renderer.${name}.log`)
    this.log(`Setup renderer logger for window ${name} to ${loggerPath}`)
    const stream = createWriteStream(loggerPath, {encoding: 'utf-8', flags: 'w+'})
    const levels = ['INFO', 'WARN', 'ERROR']
    window.webContents.on('console-message', (e, level, message, line, id) => {
      stream.write(`[${levels[level]}] [${new Date().toUTCString()}] [${id}]: ${message}\n`)
    })
    window.once('close', () => {
      window.webContents.removeAllListeners('console-message')
      stream.close()
    })
  }

  /**
   * This will create a logger prepend [${tag}] before each log from it
   * @param tag The tag to prepend
   */
  createLoggerFor(tag: string): LoggerFacade {
    function transform(tag: string) {
      return new Transform({
        transform(c, e, cb) {
          cb(undefined, `[${tag}] ${c}\n`)
        }
      })
    }

    const log = transform(tag).pipe(this.loggerEntries.log)
    const warn = transform(tag).pipe(this.loggerEntries.warn)
    const error = transform(tag).pipe(this.loggerEntries.error)

    return {
      log(message: any, ...options: any[]) {
        log.write(formatMsg(message, options))
      },
      warn(message: any, ...options: any[]) {
        warn.write(formatMsg(message, options))
      },
      error(message: any, ...options: any[]) {
        error.write(formatMsg(message, options))
      }
    }
  }
}
