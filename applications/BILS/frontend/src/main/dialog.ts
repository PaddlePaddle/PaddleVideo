import {dialog, ipcMain} from 'electron'

ipcMain.handle('dialog:showCertificateTrustDialog', (event, ...args) => {
  return dialog.showCertificateTrustDialog(args[0])
})
ipcMain.handle('dialog:showErrorBox', (event, ...args) => {
  return dialog.showErrorBox(args[0], args[1])
})
ipcMain.handle('dialog:showMessageBox', (event, ...args) => {
  return dialog.showMessageBox(args[0])
})
ipcMain.handle('dialog:showOpenDialog', (event, ...args) => {
  return dialog.showOpenDialog(args[0])
})
ipcMain.handle('dialog:showSaveDialog', (event, ...args) => {
  return dialog.showSaveDialog(args[0])
})
