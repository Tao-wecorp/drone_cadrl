pip3 install -r requirements.txt
pip3 freeze > requirements.txt

catkin_make -DPYTHON_EXECUTABLE:FILEPATH=~/.virtualenvs/baselines/bin/python

ds4rv

