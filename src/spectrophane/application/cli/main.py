import typer
from spectrophane.application.cli.lithophane import lithophane_command
#from spectrophane.application.cli.training import training_command


app = typer.Typer()

app.command(name="lithophane")(lithophane_command)
#app.command(name="training")(training_command)

def run():
    app()