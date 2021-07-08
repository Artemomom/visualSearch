import React from 'react';
import PropTypes from 'prop-types';
// import { useHistory } from 'react-router-dom';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import { isMobile } from 'react-device-detect';
import { ReactComponent as Logo } from '../../assets/icons/grid_labs_logo_by_GD_subline.svg';
// import Button from '../Button';
import styles from './TopNavigation.scss';

const useStyles = makeStyles(() => ({
  root: {
    flexGrow: 1
  },
  title: props => ({
//    flexGrow: 1,
    alignItems: 'center',
//    width: '33%',
    fontSize: '24px',
    lineHeight: '32px',
    fontWeight: 'bold',
    color: '#000000',
    display: 'block'
  }),
  nav: {
    justifyContent: 'space-between',
    background: 'white',
    color: 'black',
    width: 'auto',
    alignSelf: 'center'
  },
  appBar: {
    background: 'white'
  },
  logo: {
//    flexGrow: 1,
//    width: '33%',
    height: '32px',
    marginRight: '7px'
  },
  buttonWrapper: {
    flexGrow: 1,
    width: '33%'
  }
}));

const TopNavigation = ({ isAuth, setIsAuth }) => {
  const classes = useStyles({ isMobile });
  // const history = useHistory();
  //
  // const logout = () => {
  //   setIsAuth(false);
  //   history.push('/login');
  // };

  return (
    <div className={styles.nav}>
      <AppBar position="fixed" className={classes.appBar}>
        <Toolbar className={classes.nav}>
          <Logo className={classes.logo} />
          <Typography variant="inherit" className={classes.title}>
            Similarity search
          </Typography>
          {/*<div className={classes.buttonWrapper}>*/}
          {/*  {isAuth && <Button onClick={logout} title={'Logout'} />}*/}
          {/*</div>*/}
        </Toolbar>
      </AppBar>
    </div>
  );
};

TopNavigation.propTypes = {
  isAuth: PropTypes.bool,
  setIsAuth: PropTypes.func
};

export default TopNavigation;
