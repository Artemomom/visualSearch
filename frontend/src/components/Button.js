import React from 'react';
import PropTypes from 'prop-types';
import { makeStyles } from '@material-ui/core/styles';
import MButton from '@material-ui/core/Button';

const useStyles = makeStyles(() => ({
  button: {
    background: (props) =>
      props.type === 'simple'
        ? 'white'
        : 'linear-gradient(74deg,#fe6b59 3%,#fba320 69%,#fe8d31 100%)',
    border: '1px solid #ff8516',
    borderRadius: 20,
    padding: (props) => props.padding,
    color: (props) => (props.type === 'simple' ? 'black' : 'white'),
    fontWeight: 600
  }
}));

const Button = ({
  title,
  onClick,
  disabled,
  padding = '5px 20px',
  type = 'gradient',
  variant = 'contained'
}) => {
  const classes = useStyles({ padding, type });
  return (
    <div>
      <MButton
        onClick={onClick}
        variant={variant}
        color="primary"
        className={classes.button}
        disabled={disabled}>
        {title}
      </MButton>
    </div>
  );
};

Button.propTypes = {
  title: PropTypes.string,
  onClick: PropTypes.func,
  disabled: PropTypes.bool
};

export default Button;
