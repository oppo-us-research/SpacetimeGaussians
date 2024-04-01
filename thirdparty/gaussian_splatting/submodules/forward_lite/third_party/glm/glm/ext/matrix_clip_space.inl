namespace glm
{
	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> ortho(T left, T right, T bottom, T top)
	{
		mat<4, 4, T, defaultp> Result(static_cast<T>(1));
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = - static_cast<T>(1);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoLH_ZO(T left, T right, T bottom, T top, T z_near, T z_far)
	{
		mat<4, 4, T, defaultp> Result(1);
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = static_cast<T>(1) / (z_far - z_near);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - z_near / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoLH_NO(T left, T right, T bottom, T top, T z_near, T z_far)
	{
		mat<4, 4, T, defaultp> Result(1);
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = static_cast<T>(2) / (z_far - z_near);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - (z_far + z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH_ZO(T left, T right, T bottom, T top, T z_near, T z_far)
	{
		mat<4, 4, T, defaultp> Result(1);
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = - static_cast<T>(1) / (z_far - z_near);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - z_near / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH_NO(T left, T right, T bottom, T top, T z_near, T z_far)
	{
		mat<4, 4, T, defaultp> Result(1);
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = - static_cast<T>(2) / (z_far - z_near);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - (z_far + z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoZO(T left, T right, T bottom, T top, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return orthoLH_ZO(left, right, bottom, top, z_near, z_far);
#		else
			return orthoRH_ZO(left, right, bottom, top, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoNO(T left, T right, T bottom, T top, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return orthoLH_NO(left, right, bottom, top, z_near, z_far);
#		else
			return orthoRH_NO(left, right, bottom, top, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoLH(T left, T right, T bottom, T top, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return orthoLH_ZO(left, right, bottom, top, z_near, z_far);
#		else
			return orthoLH_NO(left, right, bottom, top, z_near, z_far);
#		endif

	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH(T left, T right, T bottom, T top, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return orthoRH_ZO(left, right, bottom, top, z_near, z_far);
#		else
			return orthoRH_NO(left, right, bottom, top, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> ortho(T left, T right, T bottom, T top, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return orthoLH_ZO(left, right, bottom, top, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return orthoLH_NO(left, right, bottom, top, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return orthoRH_ZO(left, right, bottom, top, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return orthoRH_NO(left, right, bottom, top, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumLH_ZO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
		mat<4, 4, T, defaultp> Result(0);
		Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
		Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
		Result[2][0] = -(right + left) / (right - left);
		Result[2][1] = -(top + bottom) / (top - bottom);
		Result[2][2] = farVal / (farVal - nearVal);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = -(farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumLH_NO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
		mat<4, 4, T, defaultp> Result(0);
		Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
		Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
		Result[2][0] = -(right + left) / (right - left);
		Result[2][1] = -(top + bottom) / (top - bottom);
		Result[2][2] = (farVal + nearVal) / (farVal - nearVal);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumRH_ZO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
		mat<4, 4, T, defaultp> Result(0);
		Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
		Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
		Result[2][0] = (right + left) / (right - left);
		Result[2][1] = (top + bottom) / (top - bottom);
		Result[2][2] = farVal / (nearVal - farVal);
		Result[2][3] = static_cast<T>(-1);
		Result[3][2] = -(farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumRH_NO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
		mat<4, 4, T, defaultp> Result(0);
		Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
		Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
		Result[2][0] = (right + left) / (right - left);
		Result[2][1] = (top + bottom) / (top - bottom);
		Result[2][2] = - (farVal + nearVal) / (farVal - nearVal);
		Result[2][3] = static_cast<T>(-1);
		Result[3][2] = - (static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumZO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return frustumLH_ZO(left, right, bottom, top, nearVal, farVal);
#		else
			return frustumRH_ZO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumNO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return frustumLH_NO(left, right, bottom, top, nearVal, farVal);
#		else
			return frustumRH_NO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumLH(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return frustumLH_ZO(left, right, bottom, top, nearVal, farVal);
#		else
			return frustumLH_NO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumRH(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return frustumRH_ZO(left, right, bottom, top, nearVal, farVal);
#		else
			return frustumRH_NO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustum(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return frustumLH_ZO(left, right, bottom, top, nearVal, farVal);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return frustumLH_NO(left, right, bottom, top, nearVal, farVal);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return frustumRH_ZO(left, right, bottom, top, nearVal, farVal);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return frustumRH_NO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveRH_ZO(T fov_y, T aspect, T z_near, T z_far)
	{
		assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

		T const tanHalfFovy = tan(fov_y / static_cast<T>(2));

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][2] = z_far / (z_near - z_far);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = -(z_far * z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveRH_NO(T fov_y, T aspect, T z_near, T z_far)
	{
		assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

		T const tanHalfFovy = tan(fov_y / static_cast<T>(2));

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][2] = - (z_far + z_near) / (z_far - z_near);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * z_far * z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveLH_ZO(T fov_y, T aspect, T z_near, T z_far)
	{
		assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

		T const tanHalfFovy = tan(fov_y / static_cast<T>(2));

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][2] = z_far / (z_far - z_near);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = -(z_far * z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveLH_NO(T fov_y, T aspect, T z_near, T z_far)
	{
		assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

		T const tanHalfFovy = tan(fov_y / static_cast<T>(2));

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][2] = (z_far + z_near) / (z_far - z_near);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * z_far * z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveZO(T fov_y, T aspect, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return perspectiveLH_ZO(fov_y, aspect, z_near, z_far);
#		else
			return perspectiveRH_ZO(fov_y, aspect, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveNO(T fov_y, T aspect, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return perspectiveLH_NO(fov_y, aspect, z_near, z_far);
#		else
			return perspectiveRH_NO(fov_y, aspect, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveLH(T fov_y, T aspect, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return perspectiveLH_ZO(fov_y, aspect, z_near, z_far);
#		else
			return perspectiveLH_NO(fov_y, aspect, z_near, z_far);
#		endif

	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveRH(T fov_y, T aspect, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return perspectiveRH_ZO(fov_y, aspect, z_near, z_far);
#		else
			return perspectiveRH_NO(fov_y, aspect, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspective(T fov_y, T aspect, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return perspectiveLH_ZO(fov_y, aspect, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return perspectiveLH_NO(fov_y, aspect, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return perspectiveRH_ZO(fov_y, aspect, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return perspectiveRH_NO(fov_y, aspect, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovRH_ZO(T fov, T width, T height, T z_near, T z_far)
	{
		assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));

		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = z_far / (z_near - z_far);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = -(z_far * z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovRH_NO(T fov, T width, T height, T z_near, T z_far)
	{
		assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));

		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = - (z_far + z_near) / (z_far - z_near);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * z_far * z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovLH_ZO(T fov, T width, T height, T z_near, T z_far)
	{
		assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));

		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = z_far / (z_far - z_near);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = -(z_far * z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovLH_NO(T fov, T width, T height, T z_near, T z_far)
	{
		assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));

		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = (z_far + z_near) / (z_far - z_near);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * z_far * z_near) / (z_far - z_near);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovZO(T fov, T width, T height, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return perspectiveFovLH_ZO(fov, width, height, z_near, z_far);
#		else
			return perspectiveFovRH_ZO(fov, width, height, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovNO(T fov, T width, T height, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return perspectiveFovLH_NO(fov, width, height, z_near, z_far);
#		else
			return perspectiveFovRH_NO(fov, width, height, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovLH(T fov, T width, T height, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return perspectiveFovLH_ZO(fov, width, height, z_near, z_far);
#		else
			return perspectiveFovLH_NO(fov, width, height, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovRH(T fov, T width, T height, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return perspectiveFovRH_ZO(fov, width, height, z_near, z_far);
#		else
			return perspectiveFovRH_NO(fov, width, height, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFov(T fov, T width, T height, T z_near, T z_far)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return perspectiveFovLH_ZO(fov, width, height, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return perspectiveFovLH_NO(fov, width, height, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return perspectiveFovRH_ZO(fov, width, height, z_near, z_far);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return perspectiveFovRH_NO(fov, width, height, z_near, z_far);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> infinitePerspectiveRH(T fov_y, T aspect, T z_near)
	{
		T const range = tan(fov_y / static_cast<T>(2)) * z_near;
		T const left = -range * aspect;
		T const right = range * aspect;
		T const bottom = -range;
		T const top = range;

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = (static_cast<T>(2) * z_near) / (right - left);
		Result[1][1] = (static_cast<T>(2) * z_near) / (top - bottom);
		Result[2][2] = - static_cast<T>(1);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = - static_cast<T>(2) * z_near;
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> infinitePerspectiveLH(T fov_y, T aspect, T z_near)
	{
		T const range = tan(fov_y / static_cast<T>(2)) * z_near;
		T const left = -range * aspect;
		T const right = range * aspect;
		T const bottom = -range;
		T const top = range;

		mat<4, 4, T, defaultp> Result(T(0));
		Result[0][0] = (static_cast<T>(2) * z_near) / (right - left);
		Result[1][1] = (static_cast<T>(2) * z_near) / (top - bottom);
		Result[2][2] = static_cast<T>(1);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - static_cast<T>(2) * z_near;
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> infinitePerspective(T fov_y, T aspect, T z_near)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return infinitePerspectiveLH(fov_y, aspect, z_near);
#		else
			return infinitePerspectiveRH(fov_y, aspect, z_near);
#		endif
	}

	// Infinite projection matrix: http://www.terathon.com/gdc07_lengyel.pdf
	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> tweakedInfinitePerspective(T fov_y, T aspect, T z_near, T ep)
	{
		T const range = tan(fov_y / static_cast<T>(2)) * z_near;
		T const left = -range * aspect;
		T const right = range * aspect;
		T const bottom = -range;
		T const top = range;

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = (static_cast<T>(2) * z_near) / (right - left);
		Result[1][1] = (static_cast<T>(2) * z_near) / (top - bottom);
		Result[2][2] = ep - static_cast<T>(1);
		Result[2][3] = static_cast<T>(-1);
		Result[3][2] = (ep - static_cast<T>(2)) * z_near;
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> tweakedInfinitePerspective(T fov_y, T aspect, T z_near)
	{
		return tweakedInfinitePerspective(fov_y, aspect, z_near, epsilon<T>());
	}
}//namespace glm
