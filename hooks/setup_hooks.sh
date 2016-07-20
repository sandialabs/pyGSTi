# Don't forget to run this before testing :)
# (Copies all hooks to .git/hooks (where they will actually be called))
echo Setting up hooks
cp git/* ../.git/hooks
cp -a ../test/helpers/automation_tools ../.git/hooks
echo Hook setup completed
