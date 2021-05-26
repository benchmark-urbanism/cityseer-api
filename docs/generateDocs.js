// this script stopped working and no idea why or how to debug.
const path = require('path')
const { PythonShell } = require('python-shell')

const options = {
  mode: 'text',
  pythonOptions: ['-u'],
  pythonPath: path.resolve(__dirname, '../venv/bin/python'),
  scriptPath: path.resolve(__dirname, '../venv/lib/python3.9/site-packages/yapper'),
  args: ['--config', '../.yap_config.yaml'],
}

PythonShell.run('__init__.py', options, function (err, results) {
  if (err) throw err
  console.log(results)
})
