import type Electron from 'electron'

const { shell, clipboard, ipcRenderer, dialog } = (window as any).electron as typeof Electron

export function useShell() {
  return shell
}

export function useClipboard() {
  return clipboard
}

export function useIpc() {
  return ipcRenderer
}

export function useDialog() {
  return dialog
}
