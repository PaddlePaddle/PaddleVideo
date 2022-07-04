const { writeFileSync, readFileSync } = require("fs");

function main(input) {
    const version = input('version')

    if (version) {
        const package = JSON.parse(readFileSync('./package.json'))
        package.version = version
        writeFileSync('./package.json', JSON.stringify(package, null, 2))
    }
}

function getInput(name) {
    return process.env[`INPUT_${name.replace(/ /g, '_').toUpperCase()}`] || '';
}

main(getInput);
