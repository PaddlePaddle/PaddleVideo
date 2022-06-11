window.onload = () => {
    const menu = document.querySelector('.menu')
    const menuHeight = menu.offsetHeight - parseInt(getComputedStyle(menu)['paddingTop']) - parseInt(getComputedStyle(menu)['paddingBottom'])
    menu.style.height = '0'

    openMenu = e => {
        e.preventDefault()

        menu.style.left = `${e.clientX}px`
        menu.style.top = `${e.clientY + 5}px`
        menu.style.height = `${menuHeight}px`
        menu.classList.add('is-active')

        return false
    }

    colseMenu = () => {
        menu.style.height = '0'
        menu.classList.remove('is-active')
    }

    window.onclick = () => colseMenu()
}