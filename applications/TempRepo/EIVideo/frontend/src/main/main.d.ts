/* eslint-disable no-unused-vars */

// declare electron static for static file serving

declare module '*?worker' {
  import {Worker, WorkerOptions} from 'worker_threads'
  /**
   * The helper to create the worker
   */
  export default function (options: WorkerOptions): Worker
}

declare module '/@renderer/*.html' {
  /**
   * The url of the page
   */
  const url: string
  export default url
}

declare module '/@renderer/*' {
  const noop: never
  export default noop
}

declare module '/@static/*' {
  /**
   * The path of the static file
   */
  const path: string
  export default path
}

declare module '/@preload/*' {
  /**
   * The path of the preload file
   */
  const path: string
  export default path
}

declare namespace NodeJS {
  interface Global {
    __static: string
    __windowUrls: Record<string, string>
    __preloads: Record<string, string>
    __workers: Record<string, string>
  }
}
